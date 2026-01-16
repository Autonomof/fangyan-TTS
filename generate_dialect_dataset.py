# coding=utf-8
"""
方言TTS数据集生成脚本
用于生成湖南话和河南话的TTS训练数据，供CosyVoice3微调使用

功能：
1. 读取并清洗aishell前2000行 + hunan.txt/henan.txt
2. 调用火山引擎TTS API批量合成音频
3. 输出Kaldi格式索引文件 (wav.scp, text, utt2spk, spk2utt)

使用方法：
    python generate_dialect_dataset.py --mode all      # 生成全部数据
    python generate_dialect_dataset.py --mode hunan    # 只生成湖南话
    python generate_dialect_dataset.py --mode henan    # 只生成河南话
    python generate_dialect_dataset.py --dry-run       # 只生成索引文件，不调用API

作者: Antigravity AI Assistant
日期: 2026-01-16
"""

import os
import sys
import json
import time
import uuid
import base64
import argparse
import requests
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

# 加载 .env 配置
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装 python-dotenv，尝试手动加载
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# ==================== 配置区 ====================

# 火山引擎TTS API配置（从环境变量读取）
VOLCENGINE_CONFIG = {
    "appid": os.getenv("VOLCENGINE_APPID", ""),
    "access_token": os.getenv("VOLCENGINE_ACCESS_TOKEN", ""),
    "cluster": os.getenv("VOLCENGINE_CLUSTER", "volcano_tts"),
    "host": "openspeech.bytedance.com",
    "api_url": "https://openspeech.bytedance.com/api/v1/tts"
}

# 检查必要的配置
if not VOLCENGINE_CONFIG["appid"] or not VOLCENGINE_CONFIG["access_token"]:
    print("警告: 未配置火山引擎 API 凭据！")
    print("请创建 .env 文件并配置以下环境变量：")
    print("  VOLCENGINE_APPID=your_appid")
    print("  VOLCENGINE_ACCESS_TOKEN=your_access_token")
    print("或参考 .env.example 文件")

# 音色配置
VOICE_TYPES = {
    "hunan": "BV216_streaming",   # 长沙靓女
    "henan": "BV214_streaming",   # 乡村企业家/河南话
}

# Speaker ID 配置
SPEAKER_IDS = {
    "hunan": "speaker_hunan",
    "henan": "speaker_henan",
}

# 数据配置
AISHELL_FILE = "aishell_transcript_v0.8.txt"
HUNAN_FILE = "hunan.txt"
HENAN_FILE = "henan.txt"
AISHELL_COUNT = 2000  # 提取aishell前N条
OUTPUT_DIR = "dataset"

# API调用配置
QPS_LIMIT = 2           # 每秒请求数限制
MAX_RETRIES = 3         # 最大重试次数
RETRY_DELAY = 2         # 重试间隔（秒）
REQUEST_TIMEOUT = 30    # 请求超时（秒）

# ==================== 日志配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== 数据类 ====================

@dataclass
class TextItem:
    """文本条目"""
    utt_id: str         # 语音ID
    text: str           # 文本内容
    speaker_id: str     # 说话人ID
    voice_type: str     # TTS音色
    dialect: str        # 方言类型 (hunan/henan)


# ==================== 数据加载函数 ====================

def load_aishell_data(file_path: str, count: int) -> List[str]:
    """
    加载aishell数据并清洗
    
    Args:
        file_path: aishell文件路径
        count: 提取条数
    
    Returns:
        清洗后的文本列表
    """
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= count:
                    break
                line = line.strip()
                if not line:
                    continue
                # 格式: ID text (空格分隔)
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    # 去掉文本中的空格，还原为连续汉字
                    text = parts[1].replace(' ', '')
                    if text:
                        texts.append(text)
        logger.info(f"从 {file_path} 加载了 {len(texts)} 条数据")
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    return texts


def load_dialect_data(file_path: str) -> List[str]:
    """
    加载方言文本数据
    
    Args:
        file_path: 方言文件路径
    
    Returns:
        文本列表
    """
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        logger.info(f"从 {file_path} 加载了 {len(texts)} 条方言数据")
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    return texts


def prepare_dataset(base_dir: str) -> Dict[str, List[TextItem]]:
    """
    准备完整数据集
    
    策略：
    - aishell前1000条 + hunan.txt -> speaker_hunan (使用长沙话音色)
    - aishell后1000条 + henan.txt -> speaker_henan (使用河南话音色)
    
    Args:
        base_dir: 基础目录
    
    Returns:
        按方言分组的TextItem字典
    """
    aishell_path = os.path.join(base_dir, AISHELL_FILE)
    hunan_path = os.path.join(base_dir, HUNAN_FILE)
    henan_path = os.path.join(base_dir, HENAN_FILE)
    
    # 加载数据
    aishell_texts = load_aishell_data(aishell_path, AISHELL_COUNT)
    hunan_texts = load_dialect_data(hunan_path)
    henan_texts = load_dialect_data(henan_path)
    
    # 分割aishell数据
    aishell_half = len(aishell_texts) // 2
    aishell_for_hunan = aishell_texts[:aishell_half]
    aishell_for_henan = aishell_texts[aishell_half:]
    
    dataset = {"hunan": [], "henan": []}
    
    # 构建湖南话数据集
    idx = 0
    for text in aishell_for_hunan:
        item = TextItem(
            utt_id=f"hunan_{idx:05d}",
            text=text,
            speaker_id=SPEAKER_IDS["hunan"],
            voice_type=VOICE_TYPES["hunan"],
            dialect="hunan"
        )
        dataset["hunan"].append(item)
        idx += 1
    
    for text in hunan_texts:
        item = TextItem(
            utt_id=f"hunan_{idx:05d}",
            text=text,
            speaker_id=SPEAKER_IDS["hunan"],
            voice_type=VOICE_TYPES["hunan"],
            dialect="hunan"
        )
        dataset["hunan"].append(item)
        idx += 1
    
    # 构建河南话数据集
    idx = 0
    for text in aishell_for_henan:
        item = TextItem(
            utt_id=f"henan_{idx:05d}",
            text=text,
            speaker_id=SPEAKER_IDS["henan"],
            voice_type=VOICE_TYPES["henan"],
            dialect="henan"
        )
        dataset["henan"].append(item)
        idx += 1
    
    for text in henan_texts:
        item = TextItem(
            utt_id=f"henan_{idx:05d}",
            text=text,
            speaker_id=SPEAKER_IDS["henan"],
            voice_type=VOICE_TYPES["henan"],
            dialect="henan"
        )
        dataset["henan"].append(item)
        idx += 1
    
    logger.info(f"数据集准备完成: 湖南话 {len(dataset['hunan'])} 条, 河南话 {len(dataset['henan'])} 条")
    
    return dataset


# ==================== TTS合成函数 ====================

def synthesize_single(item: TextItem, output_dir: str) -> Tuple[bool, str]:
    """
    合成单条音频
    
    Args:
        item: TextItem对象
        output_dir: 输出目录
    
    Returns:
        (成功标志, 音频文件路径或错误信息)
    """
    # 构建输出路径
    wav_dir = os.path.join(output_dir, item.dialect, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    output_path = os.path.join(wav_dir, f"{item.utt_id}.mp3")
    
    # 如果已存在则跳过
    if os.path.exists(output_path):
        return True, output_path
    
    # 构建请求
    header = {"Authorization": f"Bearer;{VOLCENGINE_CONFIG['access_token']}"}
    request_json = {
        "app": {
            "appid": VOLCENGINE_CONFIG["appid"],
            "token": "access_token",
            "cluster": VOLCENGINE_CONFIG["cluster"]
        },
        "user": {
            "uid": "dialect_dataset_generator"
        },
        "audio": {
            "voice_type": item.voice_type,
            "encoding": "mp3",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": item.text,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"
        }
    }
    
    # 重试机制
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                VOLCENGINE_CONFIG["api_url"],
                json=request_json,
                headers=header,
                timeout=REQUEST_TIMEOUT
            )
            
            result = resp.json()
            
            if "data" in result:
                audio_data = base64.b64decode(result["data"])
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                return True, output_path
            else:
                error_msg = result.get("message", "Unknown error")
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"[{item.utt_id}] API返回错误: {error_msg}, 重试 {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(RETRY_DELAY)
                else:
                    return False, f"API错误: {error_msg}"
                    
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[{item.utt_id}] 请求超时, 重试 {attempt + 1}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            else:
                return False, "请求超时"
                
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"[{item.utt_id}] 请求异常: {e}, 重试 {attempt + 1}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            else:
                return False, f"请求异常: {e}"
    
    return False, "重试次数耗尽"


def synthesize_batch(items: List[TextItem], output_dir: str, qps_limit: int = QPS_LIMIT) -> Tuple[int, int]:
    """
    批量合成音频（带限流）
    
    Args:
        items: TextItem列表
        output_dir: 输出目录
        qps_limit: 每秒请求数限制
    
    Returns:
        (成功数, 失败数)
    """
    success_count = 0
    fail_count = 0
    failed_items = []
    
    total = len(items)
    interval = 1.0 / qps_limit  # 请求间隔
    
    logger.info(f"开始合成 {total} 条音频, QPS限制: {qps_limit}")
    
    for i, item in enumerate(items):
        start_time = time.time()
        
        success, result = synthesize_single(item, output_dir)
        
        if success:
            success_count += 1
            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i + 1}/{total} ({(i + 1) * 100 // total}%), 成功: {success_count}, 失败: {fail_count}")
        else:
            fail_count += 1
            failed_items.append((item.utt_id, item.text[:30], result))
            logger.error(f"[{item.utt_id}] 合成失败: {result}")
        
        # 限流
        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
    
    # 输出失败列表
    if failed_items:
        logger.warning(f"\n失败列表 ({len(failed_items)} 条):")
        for utt_id, text_preview, error in failed_items[:10]:
            logger.warning(f"  - {utt_id}: {text_preview}... -> {error}")
        if len(failed_items) > 10:
            logger.warning(f"  ... 以及 {len(failed_items) - 10} 条更多")
    
    return success_count, fail_count


# ==================== 索引文件生成函数 ====================

def generate_kaldi_files(items: List[TextItem], output_dir: str, dialect: str):
    """
    生成Kaldi格式索引文件
    
    Args:
        items: TextItem列表
        output_dir: 输出目录
        dialect: 方言类型
    """
    dialect_dir = os.path.join(output_dir, dialect)
    os.makedirs(dialect_dir, exist_ok=True)
    
    wav_scp_path = os.path.join(dialect_dir, "wav.scp")
    text_path = os.path.join(dialect_dir, "text")
    utt2spk_path = os.path.join(dialect_dir, "utt2spk")
    spk2utt_path = os.path.join(dialect_dir, "spk2utt")
    
    # 收集spk2utt数据
    spk2utt_dict = {}
    
    with open(wav_scp_path, 'w', encoding='utf-8') as wav_f, \
         open(text_path, 'w', encoding='utf-8') as text_f, \
         open(utt2spk_path, 'w', encoding='utf-8') as utt2spk_f:
        
        for item in items:
            wav_abs_path = os.path.abspath(os.path.join(dialect_dir, "wavs", f"{item.utt_id}.mp3"))
            
            # wav.scp: utt_id /abs/path/to/wav
            wav_f.write(f"{item.utt_id} {wav_abs_path}\n")
            
            # text: utt_id text
            text_f.write(f"{item.utt_id} {item.text}\n")
            
            # utt2spk: utt_id speaker_id
            utt2spk_f.write(f"{item.utt_id} {item.speaker_id}\n")
            
            # 收集spk2utt
            if item.speaker_id not in spk2utt_dict:
                spk2utt_dict[item.speaker_id] = []
            spk2utt_dict[item.speaker_id].append(item.utt_id)
    
    # 写入spk2utt
    with open(spk2utt_path, 'w', encoding='utf-8') as spk2utt_f:
        for spk_id, utt_ids in spk2utt_dict.items():
            spk2utt_f.write(f"{spk_id} {' '.join(utt_ids)}\n")
    
    logger.info(f"[{dialect}] Kaldi索引文件已生成:")
    logger.info(f"  - {wav_scp_path}")
    logger.info(f"  - {text_path}")
    logger.info(f"  - {utt2spk_path}")
    logger.info(f"  - {spk2utt_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="方言TTS数据集生成脚本")
    parser.add_argument(
        "--mode",
        choices=["all", "hunan", "henan"],
        default="all",
        help="生成模式: all=全部, hunan=仅湖南话, henan=仅河南话"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成索引文件，不调用TTS API"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"输出目录 (默认: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--qps",
        type=int,
        default=QPS_LIMIT,
        help=f"API请求QPS限制 (默认: {QPS_LIMIT})"
    )
    
    args = parser.parse_args()
    
    qps_limit = args.qps
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    
    logger.info("=" * 60)
    logger.info("方言TTS数据集生成脚本")
    logger.info("=" * 60)
    logger.info(f"模式: {args.mode}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"QPS限制: {qps_limit}")
    logger.info("=" * 60)
    
    # 准备数据集
    try:
        dataset = prepare_dataset(script_dir)
    except FileNotFoundError:
        logger.error("数据文件加载失败，请检查文件路径")
        sys.exit(1)
    
    # 确定要处理的方言
    dialects_to_process = []
    if args.mode == "all":
        dialects_to_process = ["hunan", "henan"]
    else:
        dialects_to_process = [args.mode]
    
    # 处理每种方言
    for dialect in dialects_to_process:
        items = dataset[dialect]
        logger.info(f"\n{'=' * 40}")
        logger.info(f"处理 {dialect.upper()} 数据集 ({len(items)} 条)")
        logger.info(f"{'=' * 40}")
        
        # 生成索引文件
        generate_kaldi_files(items, output_dir, dialect)
        
        # 如果不是dry-run，则进行TTS合成
        if not args.dry_run:
            success, fail = synthesize_batch(items, output_dir, qps_limit)
            logger.info(f"\n[{dialect}] 合成完成: 成功 {success} 条, 失败 {fail} 条")
        else:
            logger.info(f"[{dialect}] Dry Run模式，跳过TTS合成")
    
    logger.info("\n" + "=" * 60)
    logger.info("全部处理完成!")
    logger.info(f"数据保存在: {output_dir}")
    logger.info("=" * 60)
    
    # 输出后续步骤提示
    print("\n" + "-" * 60)
    print("后续步骤:")
    print("-" * 60)
    print("""
1. 转换音频格式 (MP3 -> WAV 16kHz mono):
   for f in dataset/*/wavs/*.mp3; do
     ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
   done

2. 提取Speaker Embedding:
   cd CosyVoice
   python tools/extract_embedding.py --dir ../dataset/hunan \\
     --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
   python tools/extract_embedding.py --dir ../dataset/henan \\
     --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx

3. 提取Speech Token:
   python tools/extract_speech_token.py --dir ../dataset/hunan \\
     --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx
   python tools/extract_speech_token.py --dir ../dataset/henan \\
     --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx

4. 生成Parquet:
   python tools/make_parquet_list.py --num_utts_per_parquet 500 \\
     --num_processes 4 --instruct \\
     --src_dir ../dataset/hunan --des_dir ../dataset/hunan/parquet
   python tools/make_parquet_list.py --num_utts_per_parquet 500 \\
     --num_processes 4 --instruct \\
     --src_dir ../dataset/henan --des_dir ../dataset/henan/parquet

5. 开始训练:
   # 合并data.list
   cat dataset/hunan/parquet/data.list dataset/henan/parquet/data.list > dataset/train.data.list
   # 参考 CosyVoice/examples/libritts/cosyvoice3/run.sh 进行训练
""")


if __name__ == "__main__":
    main()
