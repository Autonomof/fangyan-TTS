# coding=utf-8
"""
情感语音数据集生成脚本（FunASR 版本）
使用 Paraformer 中文 ASR，生成 CosyVoice3 训练数据

特性：
- 自动 CPU / CUDA 判断
- 稳定、适合大规模离线 ASR
- 输出 Kaldi + instruct + cache
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

import torch
import soundfile as sf
from funasr import AutoModel

# ==================== 日志配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==================== 数据源配置 ====================

EMOTION_SOURCES = {
    "positive": {
        "instruct": "请用开心高兴快乐的语气说",
        "speaker": "speaker_positive",
        "sources": [
            {"path": "Chinese-Speech-Emotion-Datasets/TrainingDataWAV/Positive", "name": "csed_positive"},
            {"path": "CASIA/happy", "name": "casia_happy"},
            {"path": "CASIA/surprise", "name": "casia_surprise"},
            {"path": "CASIA/angry", "name": "casia_angry"},
            {"path": "CASIA/fear", "name": "casia_fear"},
        ],
    },
    "neutral": {
        "instruct": "",
        "speaker": "speaker_neutral",
        "sources": [
            {"path": "CASIA/neutral", "name": "casia_neutral"},
            {"path": "Chinese-Speech-Emotion-Datasets/TrainingDataWAV/Negative", "name": "csed_negative"},
        ],
    },
}

OUTPUT_DIR = "dataset_emotion"

# ==================== 数据结构 ====================

@dataclass
class AudioItem:
    utt_id: str
    audio_path: str
    text: str
    speaker_id: str
    instruct: str
    emotion_group: str


# ==================== ASR 模块（FunASR） ====================

class ASREngine:
    """FunASR Paraformer ASR"""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None

    def load_model(self):
        logger.info(f"加载 FunASR Paraformer，device={self.device}")

        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device=self.device,
        )

        logger.info("FunASR 模型加载完成")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.load_model()

        try:
            audio, sr = sf.read(audio_path)
            result = self.model.generate(
                input=audio,
                sampling_rate=sr,
                batch_size=1,
            )

            if not result:
                return ""

            text = result[0].get("text", "").strip()
            return text

        except Exception as e:
            logger.warning(f"ASR 失败 {audio_path}: {e}")
            return ""


# ==================== 扫描音频 ====================

def scan_audio_files(base_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    result = {}

    for group, cfg in EMOTION_SOURCES.items():
        result[group] = []

        for src in cfg["sources"]:
            src_path = os.path.join(base_dir, src["path"])
            if not os.path.exists(src_path):
                logger.warning(f"目录不存在: {src_path}")
                continue

            wavs = [
                (os.path.join(src_path, f), src["name"])
                for f in os.listdir(src_path)
                if f.lower().endswith(".wav")
            ]

            logger.info(f"[{group}] {src['name']} 找到 {len(wavs)} 个音频")
            result[group].extend(wavs)

    return result


# ==================== 处理音频 ====================

def process_audio_files(
    audio_files: Dict[str, List[Tuple[str, str]]],
    asr: ASREngine,
) -> Dict[str, List[AudioItem]]:

    dataset = {}

    for group, files in audio_files.items():
        cfg = EMOTION_SOURCES[group]
        dataset[group] = []

        logger.info(f"\n处理 {group} 组，共 {len(files)} 条")

        for idx, (audio_path, src_name) in enumerate(files):
            if idx % 10 == 0:
                logger.info(f"进度 {idx}/{len(files)}")

            text = asr.transcribe(audio_path)
            if not text:
                continue

            utt_id = f"{group}_{src_name}_{idx:06d}"

            dataset[group].append(
                AudioItem(
                    utt_id=utt_id,
                    audio_path=audio_path,
                    text=text,
                    speaker_id=cfg["speaker"],
                    instruct=cfg["instruct"],
                    emotion_group=group,
                )
            )

        logger.info(f"[{group}] 成功 {len(dataset[group])} 条")

    return dataset


# ==================== Kaldi 输出 ====================

def generate_kaldi_files(items: List[AudioItem], output_dir: str, group: str):
    group_dir = os.path.join(output_dir, group)
    wav_dir = os.path.join(group_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    wav_scp = open(os.path.join(group_dir, "wav.scp"), "w", encoding="utf-8")
    text_f = open(os.path.join(group_dir, "text"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(group_dir, "utt2spk"), "w", encoding="utf-8")

    spk2utt = {}

    for item in items:
        dst = os.path.join(wav_dir, f"{item.utt_id}.wav")
        if not os.path.exists(dst):
            shutil.copy2(item.audio_path, dst)

        abs_path = os.path.abspath(dst)

        wav_scp.write(f"{item.utt_id} {abs_path}\n")
        text_f.write(f"{item.utt_id} {item.text}\n")
        utt2spk.write(f"{item.utt_id} {item.speaker_id}\n")

        spk2utt.setdefault(item.speaker_id, []).append(item.utt_id)

    wav_scp.close()
    text_f.close()
    utt2spk.close()

    with open(os.path.join(group_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")

    with open(os.path.join(group_dir, "instruct.txt"), "w", encoding="utf-8") as f:
        f.write(items[0].instruct if items else "")

    logger.info(f"[{group}] Kaldi 文件生成完成")


# ==================== 缓存 ====================

def save_dataset_cache(dataset, output_dir):
    cache = {}
    for g, items in dataset.items():
        cache[g] = [item.__dict__ for item in items]

    with open(os.path.join(output_dir, "dataset_cache.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = scan_audio_files(base_dir)

    asr = ASREngine(device=args.device)
    dataset = process_audio_files(audio_files, asr)

    for group, items in dataset.items():
        if items:
            generate_kaldi_files(items, args.output_dir, group)

    save_dataset_cache(dataset, args.output_dir)

    logger.info("🎉 数据集生成完成")


if __name__ == "__main__":
    main()
