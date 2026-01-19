# coding=utf-8
"""
方言TTS数据集生成脚本
用于生成多种方言（湖南话、河南话、粤语、天津话、川渝话、郑州话、湖南普通话、东北话、西安话、上海话、广西普通话）的TTS训练数据，供CosyVoice3微调使用

功能：
1. 读取并清洗aishell前2000行 + 各方言特色文本(如hunan.txt, cantonese.txt等)
2. 调用火山引擎TTS API批量合成音频
3. 输出Kaldi格式索引文件 (wav.scp, text, utt2spk, spk2utt)

使用方法：
    python generate_dialect_dataset.py --mode all      # 生成全部数据
    python generate_dialect_dataset.py --mode cantonese # 只生成粤语
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
import random
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

# ==================== 方言配置 ====================
# Key: 方言标识
DIALECT_CONFIG = {
    "hunan": {
        "voice": "BV216_streaming",
        "speaker": "speaker_hunan",
        "text_file": "hunan.txt",
        "desc": "湖南话(长沙靓女)"
    },
    "henan": {
        "voice": "BV214_streaming",
        "speaker": "speaker_henan",
        "text_file": "henan.txt",
        "desc": "河南话(乡村企业家)"
    },
    "cantonese": {
        "voice": "BV026_streaming",
        "speaker": "speaker_cantonese",
        "text_file": "cantonese.txt",
        "desc": "粤语"
    },
    "tianjin": {
        "voice": "BV212_streaming",
        "speaker": "speaker_tianjin",
        "text_file": "tianjin.txt",
        "desc": "天津话"
    },
    "sichuan": {
        "voice": "BV019_streaming",
        "speaker": "speaker_sichuan",
        "text_file": "sichuan.txt",
        "desc": "川渝话"
    },
    "zhengzhou": {
        "voice": "BV214_streaming",
        "speaker": "speaker_zhengzhou",
        "text_file": "zhengzhou.txt",
        "desc": "郑州话"
    },
    "hunan_pu": {
        "voice": "BV226_streaming",
        "speaker": "speaker_hunan_pu",
        "text_file": "hunan_pu.txt",
        "desc": "湖南普通话"
    },
    "dongbei": {
        "voice": "BV021_streaming",
        "speaker": "speaker_dongbei",
        "text_file": "dongbei.txt",
        "desc": "东北话"
    },
    "xian": {
        "voice": "BV210_streaming",
        "speaker": "speaker_xian",
        "text_file": "xian.txt",
        "desc": "西安话"
    },
    "shanghai": {
        "voice": "BV217_streaming",
        "speaker": "speaker_shanghai",
        "text_file": "shanghai.txt",
        "desc": "上海话"
    },
    "guangxi": {
        "voice": "BV213_streaming",
        "speaker": "speaker_guangxi",
        "text_file": "guangxi.txt",
        "desc": "广西普通话"
    }
}

# 数据配置
AISHELL_FILE = "aishell_transcript_v0.8.txt"
AISHELL_PER_DIALECT_COUNT = 5000  # 每个方言随机提取N条
OUTPUT_DIR = "dataset"

# API调用配置
QPS_LIMIT = 10          # 每秒请求数限制 (默认保守值，如拥有更高配额可调整)
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
                if count > 0 and i >= count:
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


def prepare_dataset(base_dir: str, input_dir: str) -> Dict[str, List[TextItem]]:
    """
    准备完整数据集
    
    策略：
    - 将aishell数据均匀分配给所有启用的方言
    - 每个方言加上自己特定的文本数据
    
    Args:
        base_dir: 基础目录（脚本所在目录，用于fallback）
        input_dir: 输入目录（包含方言文本）
    
    Returns:
        按方言分组的TextItem字典
    """
    # AIShell文件查找逻辑：先在输入目录找，再在脚本目录找
    aishell_path = os.path.join(input_dir, AISHELL_FILE)
    if not os.path.exists(aishell_path):
        logger.info(f"在输入目录未找到 {AISHELL_FILE}，尝试在脚本目录查找...")
        aishell_path = os.path.join(base_dir, AISHELL_FILE)
    
    # 1. 加载AIShell全部数据
    logger.info(f"正在加载并清洗完整的 AIShell 数据 ({aishell_path})...")
    aishell_texts = load_aishell_data(aishell_path, -1)
    
    dataset = {}
    dialects = list(DIALECT_CONFIG.keys())
    
    logger.info(f"共有 {len(dialects)} 种方言，将为每种方言随机抽取 {AISHELL_PER_DIALECT_COUNT} 条数据")
    
    for i, dialect in enumerate(dialects):
        dataset[dialect] = []
        conf = DIALECT_CONFIG[dialect]
        
        idx = 0
        
        # A. 分配AIShell数据 (随机采样)
        # 即使数据不够，也取最大可能数量
        sample_count = min(len(aishell_texts), AISHELL_PER_DIALECT_COUNT)
        if sample_count < AISHELL_PER_DIALECT_COUNT:
            logger.warning(f"AIShell数据总量 ({len(aishell_texts)}) 不足 {AISHELL_PER_DIALECT_COUNT} 条，将使用全部数据")
        
        subset_aishell = random.sample(aishell_texts, sample_count)
        
        for text in subset_aishell:
            item = TextItem(
                utt_id=f"{dialect}_{idx:05d}",
                text=text,
                speaker_id=conf["speaker"],
                voice_type=conf["voice"],
                dialect=dialect
            )
            dataset[dialect].append(item)
            idx += 1
            
        # B. 加载方言特定数据
        # 始终优先从 input_dir 加载方言文本
        dialect_file = os.path.join(input_dir, conf["text_file"])
        try:
            d_texts = load_dialect_data(dialect_file)
            for text in d_texts:
                item = TextItem(
                    utt_id=f"{dialect}_{idx:05d}",
                    text=text,
                    speaker_id=conf["speaker"],
                    voice_type=conf["voice"],
                    dialect=dialect
                )
                dataset[dialect].append(item)
                idx += 1
            logger.info(f"[{conf['desc']}] 加载特定数据 {len(d_texts)} 条")
        except FileNotFoundError:
            # 兼容性尝试：如果在 input_dir 找不到，尝试在 base_dir 找
            backup_file = os.path.join(base_dir, conf["text_file"])
            if os.path.exists(backup_file):
                logger.warning(f"[{conf['desc']}] {conf['text_file']} 不在输入目录，但在脚本目录找到了，正在加载...")
                try:
                    d_texts = load_dialect_data(backup_file)
                    for text in d_texts:
                        item = TextItem(
                            utt_id=f"{dialect}_{idx:05d}",
                            text=text,
                            speaker_id=conf["speaker"],
                            voice_type=conf["voice"],
                            dialect=dialect
                        )
                        dataset[dialect].append(item)
                        idx += 1
                    logger.info(f"[{conf['desc']}] 加载特定数据 {len(d_texts)} 条 (从备份位置)")
                except Exception as e:
                     logger.error(f"加载 {backup_file} 出错: {e}")
            else:
                logger.warning(f"[{conf['desc']}] 未找到特定文本文件 {conf['text_file']} (查找路径: {dialect_file})，仅使用基础数据")
        except Exception as e:
            logger.error(f"加载 {dialect_file} 出错: {e}")

        logger.info(f"[{conf['desc']}] 总计 {len(dataset[dialect])} 条 (AIShell: {len(subset_aishell)})")
    
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
        return True, "SKIPPED"
    
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
    skipped_count = 0
    
    total = len(items)
    interval = 1.0 / qps_limit  # 请求间隔
    
    logger.info(f"开始合成 {total} 条音频, QPS限制: {qps_limit}")
    print(f"进度: 0/{total} [0%]", end='\r')
    
    for i, item in enumerate(items):
        start_time = time.time()
        
        success, result = synthesize_single(item, output_dir)
        
        if success:
            success_count += 1
            if result == "SKIPPED":
                skipped_count += 1
        else:
            fail_count += 1
            failed_items.append((item.utt_id, item.text[:30], result))
            # logger.error(f"[{item.utt_id}] 合成失败: {result}") # 失败时打印日志会打断进度条，改为最后汇总或仅在严重错误时打印

        # 进度条逻辑
        percent = (i + 1) * 100 // total
        bar_len = 30
        filled_len = int(bar_len * (i + 1) / total)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        
        # 使用 sys.stdout 刷新
        sys.stdout.write(f"\r[{bar}] {percent}% | {i + 1}/{total} [OK: {success_count - skipped_count}, Skip: {skipped_count}, Fail: {fail_count}]")
        sys.stdout.flush()
        
        # 限流 (如果是跳过的不需要限流)
        if result != "SKIPPED":
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

    print() # 换行
    
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
        choices=["all"] + list(DIALECT_CONFIG.keys()),
        default="all",
        help="生成模式: all=全部, 或指定某种方言 (如 hunan, cantonese 等)"
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
        "--input-dir",
        default="fangyan_text_dataset",
        help="输入目录，包含方言文本文件 (默认: fangyan_text_dataset)"
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
    output_dir = args.output_dir # 支持相对或绝对路径，如果是相对路径则相对于CWD，这里不做特殊处理，直接使用用户输入
    # 为了兼容性，如果用户输入的是相对路径，我们建议相对于执行目录。
    # 原代码逻辑是 output_dir = os.path.join(script_dir, args.output_dir)，这里改为直接使用，更灵活，或者保持原样。
    # 用户请求是 "支持指定输入输出文件夹"，通常意味着相对于当前工作目录。
    # 但为了保持一致性，如果原代码是相对于脚本目录，我们先保持，或者改为相对于CWD。
    # 一般CLI工具路径参数是相对于CWD。
    # 让我们修改为相对于CWD (直接使用 args.output_dir 和 args.input_dir)，但如果用户没有提供，默认值需要在脚本目录下找吗？
    # 默认值 "fangyan_text_dataset" 是相对路径。
    
    # 修正路径逻辑：
    # input_dir 和 output_dir 如果是相对路径，则解释为相对于当前工作目录（执行脚本的目录）。
    # 但是，为了确保脚本在任何地方运行都能找到默认文件（如果它们还在项目里），
    # 我们可以保留原来的 script_dir 拼接逻辑作为 fallback，或者简单地使用 os.path.abspath。
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    logger.info("=" * 60)
    logger.info("方言TTS数据集生成脚本")
    logger.info("=" * 60)
    logger.info(f"模式: {args.mode}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"QPS限制: {qps_limit}")
    logger.info("=" * 60)
    
    # 准备数据集
    try:
        dataset = prepare_dataset(script_dir, input_dir)
    except FileNotFoundError:
        logger.error("数据文件加载失败，请检查文件路径")
        sys.exit(1)
    
    # 确定要处理的方言
    dialects_to_process = []
    if args.mode == "all":
        dialects_to_process = list(DIALECT_CONFIG.keys())
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
   # 合并data.list
   cat dataset/*/parquet/data.list > dataset/train.data.list
   # 参考 CosyVoice/examples/libritts/cosyvoice3/run.sh 进行训练
""")


if __name__ == "__main__":
    main()
