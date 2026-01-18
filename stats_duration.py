# coding=utf-8
"""
方言数据集时长统计脚本

统计 dataset 目录下各个方言的音频总时长

使用方法：
    python stats_duration.py              # 统计所有方言
    python stats_duration.py --dialect hunan  # 只统计湖南话
    python stats_duration.py --detailed   # 显示详细信息

依赖：
    pip install mutagen   # 用于读取MP3时长
    # 或
    pip install pydub     # 备选方案

作者: Antigravity AI Assistant
日期: 2026-01-19
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

# 尝试导入音频处理库
AUDIO_LIB = None
try:
    from mutagen.mp3 import MP3
    from mutagen import MutagenError
    AUDIO_LIB = "mutagen"
except ImportError:
    try:
        from pydub import AudioSegment
        AUDIO_LIB = "pydub"
    except ImportError:
        pass

# 如果没有音频库，尝试使用 ffprobe
if not AUDIO_LIB:
    import subprocess
    import shutil
    if shutil.which("ffprobe"):
        AUDIO_LIB = "ffprobe"


@dataclass
class AudioStats:
    """音频统计结果"""
    dialect: str
    file_count: int
    total_duration_seconds: float
    average_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    failed_files: List[str]

    @property
    def total_duration_formatted(self) -> str:
        """格式化总时长为 HH:MM:SS"""
        return format_duration(self.total_duration_seconds)

    @property
    def average_duration_formatted(self) -> str:
        """格式化平均时长"""
        return f"{self.average_duration_seconds:.2f}s"

    @property
    def min_duration_formatted(self) -> str:
        return f"{self.min_duration_seconds:.2f}s"

    @property
    def max_duration_formatted(self) -> str:
        return f"{self.max_duration_seconds:.2f}s"


def format_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.ms"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def get_audio_duration_mutagen(file_path: str) -> float:
    """使用 mutagen 获取音频时长（秒）"""
    try:
        audio = MP3(file_path)
        return audio.info.length
    except MutagenError:
        return -1
    except Exception:
        return -1


def get_audio_duration_pydub(file_path: str) -> float:
    """使用 pydub 获取音频时长（秒）"""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        return -1


def get_audio_duration_ffprobe(file_path: str) -> float:
    """使用 ffprobe 获取音频时长（秒）"""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
        return -1
    except Exception:
        return -1


def get_audio_duration(file_path: str) -> float:
    """获取音频时长（秒），根据可用库选择方法"""
    if AUDIO_LIB == "mutagen":
        return get_audio_duration_mutagen(file_path)
    elif AUDIO_LIB == "pydub":
        return get_audio_duration_pydub(file_path)
    elif AUDIO_LIB == "ffprobe":
        return get_audio_duration_ffprobe(file_path)
    else:
        raise RuntimeError("没有可用的音频处理库！请安装: pip install mutagen 或 pip install pydub")


def scan_dialect_directory(dialect_dir: Path, detailed: bool = False) -> AudioStats:
    """
    扫描单个方言目录，统计音频时长
    
    Args:
        dialect_dir: 方言目录路径
        detailed: 是否显示详细进度
    
    Returns:
        AudioStats 统计结果
    """
    dialect_name = dialect_dir.name
    wavs_dir = dialect_dir / "wavs"
    
    if not wavs_dir.exists():
        return AudioStats(
            dialect=dialect_name,
            file_count=0,
            total_duration_seconds=0,
            average_duration_seconds=0,
            min_duration_seconds=0,
            max_duration_seconds=0,
            failed_files=[]
        )
    
    # 收集所有音频文件
    audio_files = list(wavs_dir.glob("*.mp3")) + list(wavs_dir.glob("*.wav"))
    
    if not audio_files:
        return AudioStats(
            dialect=dialect_name,
            file_count=0,
            total_duration_seconds=0,
            average_duration_seconds=0,
            min_duration_seconds=0,
            max_duration_seconds=0,
            failed_files=[]
        )
    
    durations = []
    failed_files = []
    
    for i, audio_file in enumerate(audio_files):
        duration = get_audio_duration(str(audio_file))
        
        if duration > 0:
            durations.append(duration)
        else:
            failed_files.append(audio_file.name)
        
        # 显示进度
        if detailed and (i + 1) % 100 == 0:
            print(f"  [{dialect_name}] 已处理 {i + 1}/{len(audio_files)} 个文件...")
    
    if not durations:
        return AudioStats(
            dialect=dialect_name,
            file_count=len(audio_files),
            total_duration_seconds=0,
            average_duration_seconds=0,
            min_duration_seconds=0,
            max_duration_seconds=0,
            failed_files=failed_files
        )
    
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    return AudioStats(
        dialect=dialect_name,
        file_count=len(durations),
        total_duration_seconds=total_duration,
        average_duration_seconds=avg_duration,
        min_duration_seconds=min_duration,
        max_duration_seconds=max_duration,
        failed_files=failed_files
    )


def print_stats_table(stats_list: List[AudioStats]):
    """打印统计表格"""
    print("\n" + "=" * 80)
    print("📊 方言数据集时长统计")
    print("=" * 80)
    
    # 表头
    print(f"{'方言':<10} {'文件数':>10} {'总时长':>15} {'平均时长':>12} {'最短':>10} {'最长':>10}")
    print("-" * 80)
    
    total_files = 0
    total_seconds = 0
    
    for stats in stats_list:
        if stats.file_count > 0:
            print(f"{stats.dialect:<10} {stats.file_count:>10} {stats.total_duration_formatted:>15} "
                  f"{stats.average_duration_formatted:>12} {stats.min_duration_formatted:>10} {stats.max_duration_formatted:>10}")
            total_files += stats.file_count
            total_seconds += stats.total_duration_seconds
        else:
            print(f"{stats.dialect:<10} {'无数据':>10}")
    
    print("-" * 80)
    
    # 汇总
    if total_files > 0:
        print(f"{'合计':<10} {total_files:>10} {format_duration(total_seconds):>15} "
              f"{total_seconds / total_files:.2f}s:>12")
    
    print("=" * 80)
    
    # 显示失败文件
    for stats in stats_list:
        if stats.failed_files:
            print(f"\n⚠️  [{stats.dialect}] {len(stats.failed_files)} 个文件读取失败:")
            for f in stats.failed_files[:5]:
                print(f"   - {f}")
            if len(stats.failed_files) > 5:
                print(f"   ... 还有 {len(stats.failed_files) - 5} 个")


def print_stats_json(stats_list: List[AudioStats]):
    """以 JSON 格式输出统计结果"""
    result = {
        "dialects": [],
        "summary": {
            "total_files": 0,
            "total_duration_seconds": 0,
            "total_duration_formatted": ""
        }
    }
    
    for stats in stats_list:
        dialect_data = {
            "name": stats.dialect,
            "file_count": stats.file_count,
            "total_duration_seconds": round(stats.total_duration_seconds, 2),
            "total_duration_formatted": stats.total_duration_formatted,
            "average_duration_seconds": round(stats.average_duration_seconds, 2),
            "min_duration_seconds": round(stats.min_duration_seconds, 2),
            "max_duration_seconds": round(stats.max_duration_seconds, 2),
            "failed_files_count": len(stats.failed_files)
        }
        result["dialects"].append(dialect_data)
        result["summary"]["total_files"] += stats.file_count
        result["summary"]["total_duration_seconds"] += stats.total_duration_seconds
    
    result["summary"]["total_duration_seconds"] = round(result["summary"]["total_duration_seconds"], 2)
    result["summary"]["total_duration_formatted"] = format_duration(result["summary"]["total_duration_seconds"])
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="方言数据集时长统计脚本")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="数据集目录路径 (默认: dataset)"
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default=None,
        help="只统计指定方言 (例如: hunan, henan)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示详细处理进度"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出"
    )
    
    args = parser.parse_args()
    
    # 检查音频库
    if not AUDIO_LIB:
        print("❌ 错误: 没有可用的音频处理库！")
        print("请安装以下任意一个：")
        print("  pip install mutagen    # 推荐，轻量快速")
        print("  pip install pydub      # 需要 ffmpeg")
        print("或确保系统已安装 ffprobe (ffmpeg 的一部分)")
        sys.exit(1)
    
    if args.detailed and not args.json:
        print(f"ℹ️  使用音频库: {AUDIO_LIB}")
    
    # 获取数据集目录
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / args.dataset_dir
    
    if not dataset_dir.exists():
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        sys.exit(1)
    
    # 确定要统计的方言
    if args.dialect:
        dialect_dirs = [dataset_dir / args.dialect]
        if not dialect_dirs[0].exists():
            print(f"❌ 错误: 方言目录不存在: {dialect_dirs[0]}")
            sys.exit(1)
    else:
        # 扫描所有子目录
        dialect_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not dialect_dirs:
        print("❌ 没有找到方言目录")
        sys.exit(1)
    
    # 统计每个方言
    stats_list = []
    for dialect_dir in sorted(dialect_dirs):
        if args.detailed and not args.json:
            print(f"📂 扫描 {dialect_dir.name}...")
        stats = scan_dialect_directory(dialect_dir, detailed=args.detailed)
        stats_list.append(stats)
    
    # 输出结果
    if args.json:
        print_stats_json(stats_list)
    else:
        print_stats_table(stats_list)


if __name__ == "__main__":
    main()
