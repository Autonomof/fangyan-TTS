# coding=utf-8
"""
方言训练数据准备脚本

功能：
1. 生成 instruct 文件（CosyVoice3 需要）
2. 将 MP3 转换为 WAV（16kHz mono）
3. 更新 wav.scp 指向 WAV 文件
4. 验证数据完整性

使用方法：
    python prepare_training_data.py --mode all        # 完整准备
    python prepare_training_data.py --mode instruct   # 只生成 instruct
    python prepare_training_data.py --mode convert    # 只转换音频
    python prepare_training_data.py --mode validate   # 验证数据

作者: Antigravity AI Assistant
日期: 2026-01-19
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置
DATASET_DIR = "dataset"
DIALECTS = ["hunan", "henan"]
SAMPLE_RATE = 16000

# 方言指令配置
INSTRUCT_TEMPLATES = {
    "hunan": "Please speak in Hunanese dialect (Changsha accent).<|endofprompt|>",
    "henan": "Please speak in Henanese dialect (Henan accent).<|endofprompt|>",
    "default": "You are a helpful assistant.<|endofprompt|>"
}


def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用"""
    return shutil.which("ffmpeg") is not None


def convert_single_audio(args: Tuple[str, str]) -> Tuple[bool, str]:
    """转换单个音频文件"""
    mp3_path, wav_path = args
    
    if os.path.exists(wav_path):
        return True, wav_path
    
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            wav_path
        ], capture_output=True, timeout=30)
        
        if result.returncode == 0:
            return True, wav_path
        else:
            return False, f"FFmpeg error: {result.stderr.decode()[:100]}"
    except Exception as e:
        return False, str(e)


def generate_instruct_file(data_dir: Path, dialect: str) -> int:
    """生成 instruct 文件"""
    text_file = data_dir / "text"
    instruct_file = data_dir / "instruct"
    
    if not text_file.exists():
        print(f"  ❌ 错误: {text_file} 不存在")
        return 0
    
    instruct_text = INSTRUCT_TEMPLATES.get(dialect, INSTRUCT_TEMPLATES["default"])
    
    count = 0
    with open(text_file, 'r', encoding='utf-8') as f_in, \
         open(instruct_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 1:
                utt_id = parts[0]
                f_out.write(f"{utt_id} {instruct_text}\n")
                count += 1
    
    print(f"  ✅ 生成 {instruct_file}, 共 {count} 条")
    return count


def convert_audio_files(data_dir: Path, num_workers: int = 4) -> Tuple[int, int]:
    """将 MP3 转换为 WAV"""
    wav_scp = data_dir / "wav.scp"
    wavs_dir = data_dir / "wavs"
    
    if not wav_scp.exists():
        print(f"  ❌ 错误: {wav_scp} 不存在")
        return 0, 0
    
    # 读取 wav.scp
    mp3_files = []
    with open(wav_scp, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt_id = parts[0]
                mp3_path = parts[1]
                wav_path = mp3_path.replace('.mp3', '.wav')
                mp3_files.append((mp3_path, wav_path))
    
    if not mp3_files:
        print("  ⚠️ 没有找到 MP3 文件")
        return 0, 0
    
    # 并行转换
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_audio, args): args for args in mp3_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  转换进度"):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    print(f"  ✅ 转换完成: 成功 {success_count}, 失败 {fail_count}")
    
    # 更新 wav.scp
    if success_count > 0:
        wav_scp_new = data_dir / "wav.scp.wav"
        with open(wav_scp, 'r', encoding='utf-8') as f_in, \
             open(wav_scp_new, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id = parts[0]
                    wav_path = parts[1].replace('.mp3', '.wav')
                    f_out.write(f"{utt_id} {wav_path}\n")
        
        # 备份原文件并替换
        wav_scp_bak = data_dir / "wav.scp.mp3.bak"
        if not wav_scp_bak.exists():
            shutil.copy(wav_scp, wav_scp_bak)
        shutil.move(wav_scp_new, wav_scp)
        print(f"  ✅ 更新 wav.scp 指向 WAV 文件")
    
    return success_count, fail_count


def validate_data(data_dir: Path) -> Dict[str, bool]:
    """验证数据完整性"""
    required_files = [
        "wav.scp",
        "text",
        "utt2spk",
        "spk2utt"
    ]
    
    optional_files = [
        "instruct",
        "utt2embedding.pt",
        "spk2embedding.pt",
        "utt2speech_token.pt"
    ]
    
    result = {}
    
    print(f"\n  必需文件:")
    for f in required_files:
        exists = (data_dir / f).exists()
        result[f] = exists
        status = "✅" if exists else "❌"
        print(f"    {status} {f}")
    
    print(f"\n  可选文件:")
    for f in optional_files:
        exists = (data_dir / f).exists()
        result[f] = exists
        status = "✅" if exists else "⚪"
        print(f"    {status} {f}")
    
    # 检查 wavs 目录
    wavs_dir = data_dir / "wavs"
    if wavs_dir.exists():
        mp3_count = len(list(wavs_dir.glob("*.mp3")))
        wav_count = len(list(wavs_dir.glob("*.wav")))
        print(f"\n  音频文件:")
        print(f"    MP3: {mp3_count} 个")
        print(f"    WAV: {wav_count} 个")
        result["wavs_mp3"] = mp3_count
        result["wavs_wav"] = wav_count
    else:
        print(f"\n  ⚠️ wavs 目录不存在")
        result["wavs_mp3"] = 0
        result["wavs_wav"] = 0
    
    # 检查 parquet 目录
    parquet_dir = data_dir / "parquet"
    if parquet_dir.exists():
        parquet_count = len(list(parquet_dir.glob("*.tar")))
        data_list = parquet_dir / "data.list"
        print(f"\n  Parquet 数据:")
        print(f"    Parquet 文件: {parquet_count} 个")
        print(f"    data.list: {'✅' if data_list.exists() else '❌'}")
        result["parquet_count"] = parquet_count
        result["data_list"] = data_list.exists()
    else:
        print(f"\n  ⚪ parquet 目录不存在 (需要运行 make_parquet_list.py)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="方言训练数据准备脚本")
    parser.add_argument(
        "--mode",
        choices=["all", "instruct", "convert", "validate"],
        default="all",
        help="运行模式: all=完整准备, instruct=只生成instruct, convert=只转换音频, validate=验证数据"
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help=f"数据集目录 (默认: {DATASET_DIR})"
    )
    parser.add_argument(
        "--dialects",
        nargs="+",
        default=DIALECTS,
        help=f"方言列表 (默认: {' '.join(DIALECTS)})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="音频转换并行数 (默认: 4)"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / args.dataset_dir
    
    print("=" * 60)
    print("方言训练数据准备脚本")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"数据集目录: {dataset_dir}")
    print(f"方言: {', '.join(args.dialects)}")
    print("=" * 60)
    
    # 检查 ffmpeg
    if args.mode in ["all", "convert"]:
        if not check_ffmpeg():
            print("\n❌ 错误: 未找到 ffmpeg，请先安装 ffmpeg")
            print("  Windows: https://ffmpeg.org/download.html")
            print("  Linux: sudo apt install ffmpeg")
            print("  macOS: brew install ffmpeg")
            sys.exit(1)
        print("\n✅ ffmpeg 已安装")
    
    # 处理每个方言
    for dialect in args.dialects:
        data_dir = dataset_dir / dialect
        
        print(f"\n{'='*40}")
        print(f"处理方言: {dialect.upper()}")
        print(f"{'='*40}")
        
        if not data_dir.exists():
            print(f"❌ 目录不存在: {data_dir}")
            continue
        
        if args.mode in ["all", "instruct"]:
            print("\n📝 生成 instruct 文件...")
            generate_instruct_file(data_dir, dialect)
        
        if args.mode in ["all", "convert"]:
            print("\n🎵 转换音频文件 (MP3 -> WAV 16kHz)...")
            convert_audio_files(data_dir, args.workers)
        
        if args.mode in ["all", "validate"]:
            print("\n🔍 验证数据完整性...")
            validate_data(data_dir)
    
    print("\n" + "=" * 60)
    print("准备完成!")
    print("=" * 60)
    
    if args.mode == "all":
        print("""
下一步操作:
1. 提取 Speaker Embedding:
   cd CosyVoice/examples/dialect
   bash run.sh  # stage=1

2. 提取 Speech Token:
   bash run.sh  # stage=2

3. 生成 Parquet:
   bash run.sh  # stage=3

4. 开始训练:
   bash run.sh  # stage=5
""")


if __name__ == "__main__":
    main()
