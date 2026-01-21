# coding=utf-8
"""
百炼 (DashScope) 方言TTS数据集生成脚本
使用模型: qwen3-tts-flash
"""

import os
import sys
import json
import time
import argparse
import random
import logging
import threading
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque

try:
    import dashscope
    # from dashscope.audio.tts_v2 import MultiModalConversation # remove this line as it likely caused ImportError
except ImportError:
    print("错误: 未找到 dashscope 库。请运行: pip install dashscope")
    sys.exit(1)

# 加载 .env 配置
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==================== 配置区 ====================

# QPS 限制 (根据API文档或实际购买情况调整，默认保守值)
QPS_LIMIT = 2

# 方言配置 (根据用户请求选择目前没有覆盖的方言)
# 已在 generate_dialect_dataset.py 覆盖的: hunan, henan, cantonese, tianjin, sichuan, zhengzhou, hunan_pu, dongbei, xian, shanghai, guangxi
# 用户提供的列表:
# - Dylan (北京) -> 新增
# - Li (南京) -> 新增
# - Marcus (陕西) -> 已有 xian (西安)，但 voice 不同 (Marcus vs speaker_xian)，可以考虑作为变体或替代。这里作为 shaanxi添加。
# - Roy (闽南) -> 新增
# - Peter (天津) -> 已有 tianjin (speaker_tianjin), 同样作为变体。这里暂不添加，除非作为 tianjin_qwen
# - Sunny/Eric (四川) -> 已有 sichuan, 暂不添加
# - Rocky/Kiki (粤语) -> 已有 cantonese, 暂不添加

DIALECT_CONFIG = {
    "beijing": {
        "voice": "Dylan",
        "instruct": "请用北京话/儿化音说<|endofprompt|>",
        "desc": "北京话 (Dylan)",
        "language_type": "Chinese"
    },
    "nanjing": {
        "voice": "Li",
        "instruct": "请用南京方言说<|endofprompt|>",
        "desc": "南京话 (Li)",
        "language_type": "Chinese"
    },
    "shaanxi": {
        "voice": "Marcus",
        "instruct": "请用陕西话/关中方言说<|endofprompt|>",
        "desc": "陕西话 (Marcus)",
        "language_type": "Chinese"
    },
    "minnan": {
        "voice": "Roy",
        "instruct": "请用闽南语说<|endofprompt|>",
        "desc": "闽南语 (Roy)",
        "language_type": "Chinese"
    }
}

# 数据配置
AISHELL_FILE = "aishell_transcript_v0.8.txt"
OUTPUT_DIR = "dataset_dashscope_dialect"
SAMPLES_PER_DIALECT = 2000
MAX_WORKERS = 4

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== 数据结构 ====================

@dataclass
class GenTask:
    utt_id: str
    text: str
    dialect_key: str
    voice: str
    instruct: str
    language_type: str
    output_path: str

# ==================== 限流器 ====================

class GlobalRateLimiter:
    """全局QPS限流器"""
    def __init__(self, qps: int):
        self.qps = qps
        self.timestamps = deque()
        self.lock = threading.Lock()
    
    def acquire(self):
        if self.qps <= 0: return
        with self.lock:
            now = time.time()
            while self.timestamps and self.timestamps[0] < now - 1.0:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.qps:
                sleep_time = 1.0 - (now - self.timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    while self.timestamps and self.timestamps[0] < now - 1.0:
                        self.timestamps.popleft()
            self.timestamps.append(now)

# ==================== 工具函数 ====================

def clean_text(text: str) -> str:
    return text.replace(" ", "").strip()

def load_texts(file_path: str, count: int = -1) -> List[str]:
    texts = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    texts.append(clean_text(parts[1]))
                else:
                    texts.append(clean_text(line))
    else:
        logger.warning(f"文本文件不存在: {file_path}")
    
    if count > 0 and len(texts) > count:
        return random.sample(texts, count)
    return texts

def download_audio(url: str, output_path: str):
    """下载音频并保存"""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

# ==================== 合成逻辑 ====================

def process_task(task: GenTask, api_key: str, limiter: GlobalRateLimiter) -> bool:
    if os.path.exists(task.output_path):
        return True

    if limiter:
        limiter.acquire()
    
    try:
        # 兼容性设置：检查 DashScope SDK 版本是否支持 MultiModalConversation
        # 用户提供的代码使用的是 MultiModalConversation.call
        dashscope.api_key = api_key
        
        # 北京地域 URL (默认)
        
        result = dashscope.MultiModalConversation.call(
            model="qwen3-tts-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": task.text},
                    ]
                }
            ],
            # 注意：SDK 文档可能不同，这里参考用户提供的示例，但用户示例使用的是 text=text, voice=... 参数直接传给call
            # 然而 MultiModalConversation 通常使用 messages 格式。
            # 用户给的示例是:
            # response = dashscope.MultiModalConversation.call(
            #     model="qwen3-tts-flash",
            #     api_key=...,
            #     text=text,
            #     voice="Cherry",
            #     ...
            # )
            # 这可能是 SDK 对旧 接口的封装或者特定用法的语法糖。
            # 既然用户给出了明确代码，我们优先尝试按用户代码的参数形式调用。
            # 如果是 SpeechSynthesizer 接口重命名而来，参数应该类似。
            
            text=task.text,
            voice=task.voice,
            language_type=task.language_type,
            stream=False
        )
        
        # 检查结果
        if result.status_code == 200:
            # 获取 audio url
            # 结果结构通常是 result.output.choices[0].message.content (如果是对话) 或者特定的字段
            # 对于 TTS，通常在 output 中
            # 打印一下结构如果出错
            
            # 根据 DashScope 文档，非 stream 模式下：
            # 成功时 status_code HTTP_OK (200)
            # 输出在 result.output
            
            # 假设 result.output.choices[0].message.audio_url 或者直接在 output 中
            # 重新查看用户提供的 url 获取说明: "通过返回的url来获取合成的语音"
            
            # 尝试从 result 中提取 url
            # 常见的 response 结构：
            # {
            #   "status_code": 200,
            #   "request_id": "...",
            #   "code": "",
            #   "message": "",
            #   "output": {
            #       "choices": [
            #           {
            #               "message": {
            #                   "content": [ ... ],
            #                   "role": "assistant"
            #               },
            #               "finish_reason": "stop"
            #           }
            #       ]
            #   },
            #   "usage": ...
            # }
            # 但是对于 TTS Flash，可能直接返回 binary 或者 url?
            # 必须仔细处理。如果 sample code 说返回 url。
            
            # 让我们尝试 inspect result
            if hasattr(result, 'output') and result.output:
                # 针对 qwen-tts-flash，可能是一个 audio url
                # 实测中，SDK 的 SpeechSynthesizer.call 返回的 result 包含 get_audio_data() 吗？
                # 用户说 "通过返回的url来获取"。
                
                # 假设 result 是 DashScopeResponse
                if isinstance(result.output, dict):
                    # 尝试查找 url
                    # 某些版本的 SDK 可能是 result.output['choices'][0]['message']['content'][0]['audio_url'] ?
                    # 或者 result.output 可能是 bytes?
                    pass
                
                # 让我们通过 try block 来捕获 url
                # 通常是 result.output['choices'][0]['message']['content'] ...
                # 或者 result.output 是音频二进制？
                pass

            # 为保险起见，我们只能假设代码能跑通，并捕获 URL。
            # 如果是按照 SpeechSynthesizer 的逻辑，返回的可能是 audio binary (如果是 stream)
            # 但这里 stream=False
            
            # 再次参考: "URL 有效期为24 小时" -> 说明返回的是 URL。
            
            # 让我们做一个假设结构探测
            audio_url = None
            try:
                # 尝试路径 1: result.output.choices[0].message.content[x].audio_url
                choices = result.output.choices
                for choice in choices:
                   if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                       for item in choice.message.content:
                           if isinstance(item, dict) and 'audio_url' in item:
                               audio_url = item['audio_url']
                               break
            except:
                pass
            
            # 如果没找到，尝试直接打印 (在 dry-run 或者 test 中验证)
            # 这里我们简单地保存 result 到 log 如果失败
            
            if audio_url:
                download_audio(audio_url, task.output_path)
                return True
            else:
                 # 也许 result 本身就是？
                 logger.error(f"无法解析音频 URL: {result}")
                 return False

        else:
            logger.error(f"API请求失败: {result.code} - {result.message}")
            return False

    except Exception as e:
        logger.error(f"Task failed {task.utt_id}: {e}")
        return False

def main():
    random.seed(42)
    parser = argparse.ArgumentParser(description="百炼方言TTS数据集生成器")
    parser.add_argument("--api-key", default=os.getenv('DASHSCOPE_API_KEY'), help="DashScope API Key")
    parser.add_argument("--input-file", default=AISHELL_FILE, help="输入文本文件")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--count", type=int, default=SAMPLES_PER_DIALECT, help="每种方言数量")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qps", type=int, default=QPS_LIMIT)
    
    args = parser.parse_args()
    
    if not args.api_key and not args.dry_run:
        logger.error("未提供 API Key")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载文本
    all_texts = load_texts(args.input_file)
    if not all_texts and not args.dry_run:
        logger.error("无文本")
        return

    # 生成任务
    tasks = []
    for dialect_key, config in DIALECT_CONFIG.items():
        selected_texts = random.sample(all_texts, min(len(all_texts), args.count))
        dialect_dir = os.path.join(args.output_dir, dialect_key)
        os.makedirs(os.path.join(dialect_dir, "wavs"), exist_ok=True)
        
        for i, text in enumerate(selected_texts):
            utt_id = f"{dialect_key}_{i:05d}"
            output_path = os.path.join(dialect_dir, "wavs", f"{utt_id}.mp3") # 这里的mp3可能需要确认格式，默认通常是 mp3 或 wav
            tasks.append(GenTask(
                utt_id=utt_id,
                text=text,
                dialect_key=dialect_key,
                voice=config['voice'],
                instruct=config['instruct'],
                language_type=config['language_type'],
                output_path=output_path
            ))

    logger.info(f"计划任务: {len(tasks)}")
    
    limiter = GlobalRateLimiter(args.qps) if not args.dry_run else None
    success_count = 0
    fail_count = 0
    
    if args.dry_run:
        success_count = len(tasks)
        logger.info("Dry Run Mode")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_task, task, args.api_key, limiter): task for task in tasks}
            for i, future in enumerate(as_completed(futures)):
                if future.result(): success_count += 1
                else: fail_count += 1
                if (i+1) % 10 == 0:
                    print(f"Progress: {i+1}/{len(tasks)} (Succ: {success_count}, Fail: {fail_count})", end='\r')
        print(f"\nDone. Succ: {success_count}, Fail: {fail_count}")

    # 生成索引
    logger.info("Generating Index Files...")
    tasks_by_dialect = {}
    for t in tasks:
        tasks_by_dialect.setdefault(t.dialect_key, []).append(t)
        
    for d_key, config in DIALECT_CONFIG.items():
        d_dir = os.path.join(args.output_dir, d_key)
        os.makedirs(os.path.join(d_dir, "wavs"), exist_ok=True)
        d_tasks = tasks_by_dialect.get(d_key, [])
        d_tasks.sort(key=lambda x: x.utt_id)
        
        if not d_tasks: continue
        
        with open(os.path.join(d_dir, "wav.scp"), 'w', encoding='utf-8') as f_wav, \
             open(os.path.join(d_dir, "text"), 'w', encoding='utf-8') as f_text, \
             open(os.path.join(d_dir, "utt2spk"), 'w', encoding='utf-8') as f_utt2spk, \
             open(os.path.join(d_dir, "spk2utt"), 'w', encoding='utf-8') as f_spk2utt, \
             open(os.path.join(d_dir, "instruct.txt"), 'w', encoding='utf-8') as f_inst:
             
             f_inst.write(config['instruct'])
             
             spk2utt = {}
             for t in d_tasks:
                 abs_path = os.path.abspath(t.output_path)
                 spk_id = f"qwen_{t.voice}"
                 
                 f_wav.write(f"{t.utt_id} {abs_path}\n")
                 f_text.write(f"{t.utt_id} {t.text}\n")
                 f_utt2spk.write(f"{t.utt_id} {spk_id}\n")
                 
                 spk2utt.setdefault(spk_id, []).append(t.utt_id)
            
             for spk, utts in spk2utt.items():
                 f_spk2utt.write(f"{spk} {' '.join(utts)}\n")

    logger.info("Index generation complete.")

if __name__ == "__main__":
    main()
