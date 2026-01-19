#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper


def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt], backend='soundfile')
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    # Convert audio to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        # GPU 串行，避免多线程叠加显存峰值
        with gpu_semaphore:
            outputs = ort_session.run(
                None,
                {
                    ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
                }
            )
        speech_token = outputs[0].flatten().tolist()
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    for future in tqdm(as_completed(all_task)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    # SessionOptions：限制内部并行，降低显存峰值
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    option.inter_op_num_threads = 1

    # CUDA EP 选项：禁用 CUDA Graph，限制显存与工作区
    cuda_provider_options = {
        "enable_cuda_graph": 0,
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 可按需调整
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_use_max_workspace": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": 1,
    }
    providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    # 仅串行化 GPU 调用；CPU 侧预处理仍并发
    gpu_semaphore = threading.Semaphore(1)

    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    main(args)
