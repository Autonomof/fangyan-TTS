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
import threading
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt])
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)

    # 串行保护：同一时刻只允许一个 GPU 推理，避免显存峰值叠加
    with gpu_semaphore:
        outputs = ort_session.run(
            None,
            {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
        )
    embedding = outputs[0].flatten().tolist()
    return utt, embedding


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2embedding, spk2embedding = {}, {}
    for future in tqdm(as_completed(all_task)):
        utt, embedding = future.result()
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/utt2spk'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]

    # SessionOptions：避免内部并行也推高显存
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    option.inter_op_num_threads = 1

    # CUDA EP 选项：关闭 CUDA Graph，限制显存与工作区，减少多流拷贝问题
    cuda_provider_options = {
        "enable_cuda_graph": 0,                    # 解决"stream is capturing"
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,   # 按需调整，例如 2GB
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_use_max_workspace": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": 1,
    }
    providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    # 仅对 GPU 推理限流（设为 1 保证串行）。如显存充裕可尝试升到 2。
    gpu_semaphore = threading.Semaphore(1)

    # 保留 CPU 侧并发（读取音频、计算 fbank 等），但 GPU 有信号量保护
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
