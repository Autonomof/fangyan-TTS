# coding=utf-8
"""
方言训练数据准备脚本

功能：
1. 生成 instruct 文件（CosyVoice3 需要，使用中文指令）
2. 将 MP3 转换为 WAV（16kHz mono）【可选】
3. 合并所有方言数据到一个文件夹，方便统一训练
4. 验证数据完整性

使用方法：
    python prepare_training_data.py --mode all        # 完整准备
    python prepare_training_data.py --mode instruct   # 只生成 instruct
    python prepare_training_data.py --mode combine    # 只合并数据
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
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置
DATASET_DIR = "dataset"
COMBINED_DIR = "combined"  # 合并后的目录名
SAMPLE_RATE = 16000

# 方言指令配置（中文）
INSTRUCT_TEMPLATES = {
    # 北方方言
    "dongbei": "请用东北话说。<|endofprompt|>",
    "tianjin": "请用天津话说。<|endofprompt|>",
    "xian": "请用西安话说。<|endofprompt|>",
    "henan": "请用河南话说。<|endofprompt|>",
    "zhengzhou": "请用郑州话说。<|endofprompt|>",
    
    # 西南官话
    "sichuan": "请用四川话说。<|endofprompt|>",
    "chuanyu": "请用川渝方言说。<|endofprompt|>",
    "chongqing": "请用重庆话说。<|endofprompt|>",
    
    # 湘语
    "hunan": "请用湖南话说。<|endofprompt|>",
    "changsha": "请用长沙话说。<|endofprompt|>",
    "hunan_pu": "请用湖南普通话说。<|endofprompt|>",
    
    # 粤语
    "cantonese": "请用粤语说。<|endofprompt|>",
    "yueyu": "请用粤语说。<|endofprompt|>",
    "guangxi": "请用广西话说。<|endofprompt|>",
    
    # 吴语
    "shanghai": "请用上海话说。<|endofprompt|>",
    
    # 情感（emotion 作为特殊方言处理）
    "emotion": "请用普通话说。<|endofprompt|>",  # 情感数据已有逐句instruct，此为fallback
    
    # 默认
    "default": "请用方言说。<|endofprompt|>"
}

# 方言中文名称映射
DIALECT_NAMES = {
    "dongbei": "东北话",
    "tianjin": "天津话",
    "xian": "西安话",
    "henan": "河南话",
    "zhengzhou": "郑州话",
    "sichuan": "四川话",
    "chuanyu": "川渝方言",
    "chongqing": "重庆话",
    "hunan": "湖南话",
    "changsha": "长沙话",
    "hunan_pu": "湖南普通话",
    "cantonese": "粤语",
    "yueyu": "粤语",
    "guangxi": "广西话",
    "shanghai": "上海话",
    "emotion": "情感数据",
}


def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用"""
    return shutil.which("ffmpeg") is not None


def convert_single_audio(args: Tuple[str, str]) -> Tuple[bool, str]:
    """转换单个音频文件"""
    src_path, dst_path = args
    
    if os.path.exists(dst_path):
        return True, dst_path
    
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", src_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            dst_path
        ], capture_output=True, timeout=30)
        
        if result.returncode == 0:
            return True, dst_path
        else:
            return False, f"FFmpeg error: {result.stderr.decode()[:100]}"
    except Exception as e:
        return False, str(e)


def generate_instruct_file(data_dir: Path, dialect: str, force: bool = False) -> int:
    """
    生成 instruct 文件（中文指令）
    
    Args:
        data_dir: 数据目录
        dialect: 方言名称
        force: 是否强制覆盖已有的 instruct 文件
    
    Returns:
        生成的条目数
    """
    text_file = data_dir / "text"
    instruct_file = data_dir / "instruct"
    # 也检查 instruct.txt（ESD数据集格式）
    instruct_txt_file = data_dir / "instruct.txt"
    
    if not text_file.exists():
        print(f"  ❌ 错误: {text_file} 不存在")
        return 0
    
    # 如果已存在 instruct 或 instruct.txt，且不强制覆盖，则跳过
    if not force:
        if instruct_file.exists():
            with open(instruct_file, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"  ⏭️ 跳过: instruct 已存在 ({count} 条)")
            return count
        if instruct_txt_file.exists():
            # 将 instruct.txt 复制为 instruct（统一格式）
            import shutil
            shutil.copy(instruct_txt_file, instruct_file)
            with open(instruct_file, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"  ✅ 复制 instruct.txt -> instruct ({count} 条)")
            return count
    
    instruct_text = INSTRUCT_TEMPLATES.get(dialect, INSTRUCT_TEMPLATES["default"])
    dialect_name = DIALECT_NAMES.get(dialect, dialect)
    
    count = 0
    with open(text_file, 'r', encoding='utf-8') as f_in, \
         open(instruct_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 1:
                utt_id = parts[0]
                f_out.write(f"{utt_id} {instruct_text}\n")
                count += 1
    
    print(f"  ✅ 生成 instruct: {count} 条 (指令: {instruct_text[:20]}...)")
    return count


def combine_dialect_data(
    dataset_dir: Path, 
    dialects: List[str], 
    combined_dir: Path,
    extra_dirs: Optional[List[Path]] = None
) -> Dict[str, int]:
    """
    合并所有方言数据到一个文件夹
    
    Args:
        dataset_dir: 主数据集目录
        dialects: 方言列表（相对于 dataset_dir）
        combined_dir: 输出的合并目录
        extra_dirs: 额外的数据目录列表（绝对路径，直接包含 wav.scp 等文件）
    
    生成的文件:
    - wav.scp: 合并的音频路径索引
    - text: 合并的文本
    - utt2spk: 语音到说话人映射
    - spk2utt: 说话人到语音映射
    - instruct: 合并的指令
    """
    print(f"\n📦 合并方言数据到: {combined_dir}")
    
    # 创建合并目录
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化合并文件
    files_to_merge = ["wav.scp", "text", "utt2spk", "instruct"]
    merged_data = {f: [] for f in files_to_merge}
    spk2utt_data = {}  # 需要特殊处理
    
    stats = {
        "total_utts": 0,
        "total_speakers": 0,
        "dialects_processed": 0
    }
    
    for dialect in dialects:
        data_dir = dataset_dir / dialect
        
        # 跳过 combined 目录，避免循环引用
        if dialect == COMBINED_DIR or dialect == "combined":
            print(f"  ⏭️ 跳过 combined 目录")
            continue
        
        if not data_dir.exists():
            print(f"  ⚠️ 跳过不存在的目录: {dialect}")
            continue
        
        # 检查必需文件
        if not (data_dir / "text").exists():
            print(f"  ⚠️ 跳过 {dialect}: 缺少 text 文件")
            continue
        
        print(f"  📂 处理 {dialect}...")
        dialect_utt_count = 0
        
        for filename in files_to_merge:
            file_path = data_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            merged_data[filename].append(line)
                            if filename == "text":
                                dialect_utt_count += 1
        
        # 处理 spk2utt
        spk2utt_file = data_dir / "spk2utt"
        if spk2utt_file.exists():
            with open(spk2utt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        spk = parts[0]
                        utts = parts[1:]
                        if spk not in spk2utt_data:
                            spk2utt_data[spk] = []
                        spk2utt_data[spk].extend(utts)
        
        stats["total_utts"] += dialect_utt_count
        stats["dialects_processed"] += 1
        print(f"     语音数: {dialect_utt_count}")
    
    # 写入合并文件
    print("\n  📝 写入合并文件...")
    
    for filename, lines in merged_data.items():
        if lines:
            output_file = combined_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"     {filename}: {len(lines)} 行")
    
    # 写入 spk2utt
    if spk2utt_data:
        spk2utt_file = combined_dir / "spk2utt"
        with open(spk2utt_file, 'w', encoding='utf-8') as f:
            for spk, utts in spk2utt_data.items():
                f.write(f"{spk} {' '.join(utts)}\n")
        stats["total_speakers"] = len(spk2utt_data)
        print(f"     spk2utt: {len(spk2utt_data)} 个说话人")
    
    # 处理额外目录
    if extra_dirs:
        print(f"\n  📂 处理额外数据目录...")
        for extra_dir in extra_dirs:
            extra_path = Path(extra_dir)
            if not extra_path.exists():
                print(f"  ⚠️ 跳过不存在的目录: {extra_dir}")
                continue
            
            # 跳过 combined 目录，避免循环引用
            if extra_path.name == COMBINED_DIR or extra_path.name == "combined":
                print(f"  ⏭️ 跳过 combined 目录: {extra_path}")
                continue
            
            # 检查必需文件
            if not (extra_path / "text").exists():
                print(f"  ⚠️ 跳过 {extra_path.name}: 缺少 text 文件")
                continue
            
            print(f"  📂 处理 {extra_path.name}...")
            extra_utt_count = 0
            
            for filename in files_to_merge:
                file_path = extra_path / filename
                # 也检查 .txt 后缀版本
                if not file_path.exists():
                    file_path = extra_path / f"{filename}.txt"
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                merged_data[filename].append(line)
                                if filename == "text":
                                    extra_utt_count += 1
            
            # 处理 spk2utt
            spk2utt_file = extra_path / "spk2utt"
            if spk2utt_file.exists():
                with open(spk2utt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            spk = parts[0]
                            utts = parts[1:]
                            if spk not in spk2utt_data:
                                spk2utt_data[spk] = []
                            spk2utt_data[spk].extend(utts)
            
            stats["total_utts"] += extra_utt_count
            stats["dialects_processed"] += 1
            print(f"     语音数: {extra_utt_count}")
    
    # 重新写入合并文件（包含额外目录的数据）
    print("\n  📝 写入合并文件...")
    
    for filename, lines in merged_data.items():
        if lines:
            output_file = combined_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"     {filename}: {len(lines)} 行")
    
    # 重新写入 spk2utt
    if spk2utt_data:
        spk2utt_file = combined_dir / "spk2utt"
        with open(spk2utt_file, 'w', encoding='utf-8') as f:
            for spk in sorted(spk2utt_data.keys()):
                utts = spk2utt_data[spk]
                f.write(f"{spk} {' '.join(utts)}\n")
        stats["total_speakers"] = len(spk2utt_data)
        print(f"     spk2utt: {len(spk2utt_data)} 个说话人")
    
    print(f"\n  ✅ 合并完成!")
    print(f"     方言数: {stats['dialects_processed']}")
    print(f"     语音总数: {stats['total_utts']}")
    print(f"     说话人数: {stats['total_speakers']}")
    
    return stats


def convert_audio_files(data_dir: Path, num_workers: int = 4) -> Tuple[int, int]:
    """将 MP3 转换为 WAV"""
    wav_scp = data_dir / "wav.scp"
    
    if not wav_scp.exists():
        print(f"  ❌ 错误: {wav_scp} 不存在")
        return 0, 0
    
    # 读取 wav.scp
    audio_files = []
    with open(wav_scp, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt_id = parts[0]
                src_path = parts[1]
                if src_path.endswith('.mp3'):
                    wav_path = src_path.replace('.mp3', '.wav')
                    audio_files.append((src_path, wav_path))
    
    if not audio_files:
        print("  ⚠️ 没有需要转换的 MP3 文件")
        return 0, 0
    
    # 并行转换
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_audio, args): args for args in audio_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  转换进度"):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    print(f"  ✅ 转换完成: 成功 {success_count}, 失败 {fail_count}")
    
    # 更新 wav.scp
    if success_count > 0:
        wav_scp_new = data_dir / "wav.scp.new"
        with open(wav_scp, 'r', encoding='utf-8') as f_in, \
             open(wav_scp_new, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id = parts[0]
                    wav_path = parts[1].replace('.mp3', '.wav')
                    f_out.write(f"{utt_id} {wav_path}\n")
        
        # 备份原文件并替换
        wav_scp_bak = data_dir / "wav.scp.bak"
        if not wav_scp_bak.exists():
            shutil.copy(wav_scp, wav_scp_bak)
        shutil.move(wav_scp_new, wav_scp)
        print(f"  ✅ 更新 wav.scp 指向 WAV 文件")
    
    return success_count, fail_count


def validate_data(data_dir: Path, name: str = "") -> Dict[str, any]:
    """验证数据完整性"""
    required_files = ["wav.scp", "text", "utt2spk", "spk2utt"]
    optional_files = ["instruct", "utt2embedding.pt", "spk2embedding.pt", "utt2speech_token.pt"]
    
    result = {"name": name, "valid": True}
    
    print(f"\n  📁 {name or data_dir.name}")
    print(f"  必需文件:")
    
    for f in required_files:
        exists = (data_dir / f).exists()
        result[f] = exists
        if not exists:
            result["valid"] = False
        status = "✅" if exists else "❌"
        
        # 统计行数
        if exists:
            with open(data_dir / f, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            print(f"    {status} {f} ({line_count} 行)")
        else:
            print(f"    {status} {f}")
    
    print(f"  可选文件:")
    for f in optional_files:
        exists = (data_dir / f).exists()
        result[f] = exists
        status = "✅" if exists else "⚪"
        print(f"    {status} {f}")
    
    return result


def get_all_dialects(dataset_dir: Path) -> List[str]:
    """获取所有方言目录"""
    dialects = []
    if dataset_dir.exists():
        for item in dataset_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != COMBINED_DIR:
                # 检查是否包含 text 文件
                if (item / "text").exists():
                    dialects.append(item.name)
    return sorted(dialects)


def main():
    parser = argparse.ArgumentParser(description="方言训练数据准备脚本")
    parser.add_argument(
        "--mode",
        choices=["all", "instruct", "combine", "convert", "validate"],
        default="all",
        help="运行模式: all=完整准备, instruct=生成instruct, combine=合并数据, convert=转换音频, validate=验证"
    )
    parser.add_argument(
        "--dataset-dir",
        default=DATASET_DIR,
        help=f"数据集目录 (默认: {DATASET_DIR})"
    )
    parser.add_argument(
        "--dialects",
        nargs="*",
        default=None,
        help="方言列表，留空则自动检测所有方言"
    )
    parser.add_argument(
        "--combined-name",
        default=COMBINED_DIR,
        help=f"合并目录名 (默认: {COMBINED_DIR})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="音频转换并行数 (默认: 4)"
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="跳过音频格式转换（直接使用 MP3）"
    )
    parser.add_argument(
        "--extra-dirs",
        nargs="*",
        default=[],
        help="额外的数据目录（如 dataset_emotion），会被合并到 combined 中"
    )
    parser.add_argument(
        "--force-instruct",
        action="store_true",
        help="强制重新生成 instruct 文件（即使已存在）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="合并输出目录的绝对路径（可选，默认为 dataset-dir/combined-name）"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / args.dataset_dir
    
    # 确定合并输出目录
    if args.output_dir:
        # 使用用户指定的绝对路径
        combined_dir = Path(args.output_dir)
    else:
        # 默认为 dataset-dir/combined-name
        combined_dir = dataset_dir / args.combined_name
    
    # 自动检测方言
    if args.dialects is None or len(args.dialects) == 0:
        dialects = get_all_dialects(dataset_dir)
    else:
        dialects = args.dialects
    
    print("=" * 60)
    print("🗣️  方言训练数据准备脚本 v2.0")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"数据集目录: {dataset_dir}")
    print(f"检测到方言: {len(dialects)} 个")
    for d in dialects:
        name = DIALECT_NAMES.get(d, d)
        print(f"  - {d} ({name})")
    print(f"合并目录: {combined_dir}")
    print("=" * 60)
    
    if not dialects:
        print("❌ 没有找到任何方言数据目录")
        sys.exit(1)
    
    # ==================== 生成 instruct ====================
    if args.mode in ["all", "instruct"]:
        print("\n" + "=" * 40)
        print("📝 生成 instruct 文件（中文指令）")
        print("=" * 40)
        
        for dialect in dialects:
            data_dir = dataset_dir / dialect
            if data_dir.exists():
                print(f"\n处理 {dialect} ({DIALECT_NAMES.get(dialect, dialect)}):")
                generate_instruct_file(data_dir, dialect, force=args.force_instruct)
        
        # 也处理额外目录的 instruct
        if args.extra_dirs:
            for extra_dir in args.extra_dirs:
                extra_path = script_dir / extra_dir
                if extra_path.exists():
                    print(f"\n处理额外目录 {extra_path.name}:")
                    generate_instruct_file(extra_path, extra_path.name, force=args.force_instruct)
    
    # ==================== 音频转换 ====================
    if args.mode in ["all", "convert"] and not args.no_convert:
        print("\n" + "=" * 40)
        print("🎵 转换音频文件 (MP3 -> WAV)")
        print("=" * 40)
        
        if not check_ffmpeg():
            print("\n⚠️ 未找到 ffmpeg，跳过音频转换")
            print("  如需转换，请安装 ffmpeg")
        else:
            for dialect in dialects:
                data_dir = dataset_dir / dialect
                if data_dir.exists():
                    print(f"\n处理 {dialect}:")
                    convert_audio_files(data_dir, args.workers)
    
    # ==================== 合并数据 ====================
    if args.mode in ["all", "combine"]:
        print("\n" + "=" * 40)
        print("📦 合并所有方言数据")
        print("=" * 40)
        
        # 解析额外目录为绝对路径
        extra_paths = []
        if args.extra_dirs:
            for extra_dir in args.extra_dirs:
                extra_path = script_dir / extra_dir
                if extra_path.exists():
                    extra_paths.append(extra_path)
                else:
                    print(f"  ⚠️ 额外目录不存在: {extra_dir}")
        
        combine_dialect_data(dataset_dir, dialects, combined_dir, extra_dirs=extra_paths)
    
    # ==================== 验证数据 ====================
    if args.mode in ["all", "validate"]:
        print("\n" + "=" * 40)
        print("🔍 验证数据完整性")
        print("=" * 40)
        
        # 验证各方言
        for dialect in dialects:
            data_dir = dataset_dir / dialect
            if data_dir.exists():
                validate_data(data_dir, DIALECT_NAMES.get(dialect, dialect))
        
        # 验证合并目录
        if combined_dir.exists():
            print("\n" + "-" * 30)
            validate_data(combined_dir, "合并数据 (combined)")
    
    print("\n" + "=" * 60)
    print("✅ 准备完成!")
    print("=" * 60)
    
    if args.mode == "all":
        print(f"""
下一步操作:

1. 进入训练目录:
   cd CosyVoice/examples/dialect

2. 修改 run.sh 中的数据目录指向合并数据:
   data_dir=../../../dataset/{args.combined_name}

3. 按阶段执行训练:
   # Stage 1: 提取 Speaker Embedding
   # Stage 2: 提取 Speech Token
   # Stage 3: 生成 Parquet
   # Stage 5: 训练模型
   bash run.sh

合并数据位置: {combined_dir}
""")


if __name__ == "__main__":
    main()
