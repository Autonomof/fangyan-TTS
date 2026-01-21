# coding=utf-8
"""
阿里云情感语音数据集生成脚本
用于生成自然语言控制的情感语音数据集 (Distill CosyVoice)
参考文档: https://help.aliyun.com/zh/isi/developer-reference/ssml-overview

功能：
1. 读取文本数据 (Aishell 或 自定义文本)
2. 使用阿里云 NLS SDK (SpeechSynthesizer) 结合 SSML 生成多情感音频
3. 生成对应的自然语言指令 (Instruct)
4. 输出 Kaldi 格式 (wav.scp, text, utt2spk, spk2utt, instruct.txt)

前置要求:
    pip install alibabacloud-nls-python-sdk>=1.0.0
    pip install aliyun-python-sdk-core  #用于自动获取Token
    (请根据官方文档安装最新版SDK)

使用方法:
    1. 使用已有 Token:
       python generate_emotional_dataset_aliyun.py --appkey <APPKEY> --token <TOKEN>
    
    2. 使用 AK/SK 自动获取 Token:
       python generate_emotional_dataset_aliyun.py --appkey <APPKEY> --ak-id <ACCESS_KEY_ID> --ak-secret <ACCESS_KEY_SECRET>
"""

import os
import sys
import json
import time
import uuid
import argparse
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from dataclasses import dataclass

# 加载 .env 配置
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 尝试手动加载 .env
    from pathlib import Path
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip()

# 尝试导入 nls 库
try:
    import nls
except ImportError:
    print("错误: 未找到 alibabacloud-nls-python-sdk。请运行: pip install alibabacloud-nls-python-sdk")
    # Mock for checking syntax or if lib is missing during dev
    class nls:
        class SpeechSynthesizer:
            def __init__(self, *args, **kwargs): pass
            def start(self, *args, **kwargs): pass
            def wait_completed(self, *args, **kwargs): pass

# 尝试导入 aliyunsdkcore (用于获取Token)
try:
    from aliyunsdkcore.client import AcsClient
    from aliyunsdkcore.request import CommonRequest
except ImportError:
    AcsClient = None
    CommonRequest = None

# ==================== 配置区 ====================

# ==================== 配置区 ====================

# 支持多情感的音色列表
# 所有下列音色都支持 6 种基础情感: angry, fear, happy, neutral, sad, surprise
VOICE_POOL = [
    "zhifeng_emo", "zhibing_emo", "zhimiao_emo", 
    "zhimi_emo", "zhiyan_emo", "zhibei_emo", "zhitian_emo"
]

EMOTION_CONFIG = {
    "happy": {
        "ssml_attr": {'category': 'happy', 'intensity': '1.0'},
        "instruct": "请用开心高兴的语气说<|endofprompt|>",
        "desc": "开心"
    },
    "sad": {
        "ssml_attr": {'category': 'sad', 'intensity': '1.0'},
        "instruct": "请用悲伤难过的语气说<|endofprompt|>",
        "desc": "悲伤"
    },
    "angry": {
        "ssml_attr": {'category': 'angry', 'intensity': '1.0'},
        "instruct": "请用愤怒生气的语气说<|endofprompt|>",
        "desc": "愤怒"
    },
    "surprise": {
        "ssml_attr": {'category': 'surprise', 'intensity': '1.0'},
        "instruct": "请用惊讶吃惊的语气说<|endofprompt|>",
        "desc": "惊讶"
    },
    "fear": {
        "ssml_attr": {'category': 'fear', 'intensity': '1.0'},
        "instruct": "请用害怕恐惧的语气说<|endofprompt|>",
        "desc": "恐惧"
    },
    "neutral": {
        "ssml_attr": {'category': 'neutral', 'intensity': '1.0'},
        "instruct": "请用平时正常的语气说<|endofprompt|>",
        "desc": "中立"
    }
}

# 默认全局配置
OUTPUT_DIR = "dataset_aliyun_emotion"
AISHELL_FILE = "aishell_transcript_v0.8.txt"
SAMPLES_PER_EMOTION = 2000  # 每种情感生成的句子数量
MAX_WORKERS = 4            # 并发线程数

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
    emotion_key: str
    voice: str
    instruct: str
    output_path: str
    
# ==================== Token 获取逻辑 ====================

def fetch_token(ak_id: str, ak_secret: str, region: str = "cn-shanghai") -> str:
    """如果不提供Token但提供了AK/SK，自动获取Token"""
    if not AcsClient:
        logger.error("未找到 aliyunsdkcore，无法自动获取Token。请安装: pip install aliyun-python-sdk-core")
        return None

    logger.info("正在使用 AK/SK 获取 NLS Token...")
    client = AcsClient(ak_id, ak_secret, region)

    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')

    try:
        response = client.do_action_with_exception(request)
        jss = json.loads(response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token = jss['Token']['Id']
            expire_time = jss['Token']['ExpireTime']
            logger.info(f"Token获取成功! 过期时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expire_time))}")
            return token
        else:
            logger.error(f"获取Token失败，响应: {jss}")
    except Exception as e:
        logger.error(f"获取Token异常: {e}")
    
    return None


# ==================== 阿里云 NLS 回调类 ====================

class TtsCallback:
    def __init__(self, output_path):
        self.output_path = output_path
        self.f = None
        self.error_msg = None
        self.success = False
        self._done = threading.Event()

    def on_metainfo(self, message, *args):
        # logger.debug(f"Meta info: {message}")
        pass

    def on_error(self, message, *args):
        logger.error(f"TTS Error: {message}")
        self.error_msg = message
        self._done.set()

    def on_close(self, *args):
        # logger.debug("Connection closed")
        if self.f:
            self.f.close()
            self.f = None
        self.success = True
        self._done.set()

    def on_data(self, data, *args):
        if self.f is None:
            self.f = open(self.output_path, "wb")
        self.f.write(data)

    def on_completed(self, message, *args):
        # logger.debug(f"Completed: {message}")
        self._done.set()
        
    def wait(self):
        self._done.wait()

# ==================== 工具函数 ====================

def clean_text(text: str) -> str:
    """清理文本，去除空格等"""
    return text.replace(" ", "").strip()

def build_ssml(text: str, emotion_key: str) -> str:
    """构建 SSML 字符串"""
    config = EMOTION_CONFIG.get(emotion_key)
    if not config:
        return text # Fallback
    
    attr = config['ssml_attr']
    category = attr['category']
    intensity = attr['intensity']
    
    ssml = f'<speak><emotion category="{category}" intensity="{intensity}">{text}</emotion></speak>'
    return ssml

def load_texts(file_path: str, count: int = -1) -> List[str]:
    """加载文本数据"""
    texts = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # 假设格式: ID Text 
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    texts.append(clean_text(parts[1]))
                else:
                    texts.append(clean_text(line)) # 纯文本格式
    else:
        logger.warning(f"文本文件不存在: {file_path}")
    
    if count > 0 and len(texts) > count:
        return random.sample(texts, count)
    return texts

# ==================== 合成核心逻辑 ====================

def process_task(task: GenTask, appkey: str, token: str) -> bool:
    """处理单个合成任务"""
    if os.path.exists(task.output_path):
        # logger.info(f"Skip existing: {task.output_path}")
        return True

    callback = TtsCallback(task.output_path)
    
    # 初始化 Synthesizer
    synthesizer = nls.NlsSpeechSynthesizer(
        url="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
        token=token,
        appkey=appkey,
        on_metainfo=callback.on_metainfo,
        on_data=callback.on_data,
        on_completed=callback.on_completed,
        on_error=callback.on_error,
        on_close=callback.on_close,
        callback_args=[]
    )
    
    try:
        ssml_text = build_ssml(task.text, task.emotion_key)
        
        synthesizer.start(
            text=ssml_text,
            voice=task.voice,
            aformat="mp3",    # 生成 MP3，后续统一转 wav
            sample_rate=16000,
            volume=50,
            speech_rate=0,
            pitch_rate=0
        )
        
        # 等待完成
        callback.wait()
        
        if callback.error_msg:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Task failed {task.utt_id}: {e}")
        return False

# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="阿里云情感语音生成器")
    parser.add_argument("--appkey", default=os.getenv('ALIYUN_APPKEY'), help="阿里云 AppKey (默认从环境变量 ALIYUN_APPKEY 获取)")
    parser.add_argument("--token", help="阿里云 AccessToken (可选，未提供则尝试使用 AK/SK 获取)")
    parser.add_argument("--ak-id", help="阿里云 AccessKey ID (用于自动获取Token)")
    parser.add_argument("--ak-secret", help="阿里云 AccessKey Secret (用于自动获取Token)")
    parser.add_argument("--input-file", default=AISHELL_FILE, help="输入文本文件路径")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--count", type=int, default=SAMPLES_PER_EMOTION, help="每种情感生成的句子数")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    parser.add_argument("--dry-run", action="store_true", help="仅生成文件列表，不调用TTS API")
    
    args = parser.parse_args()

    if not args.appkey:
        logger.error("未提供 AppKey (参数 --appkey 或 环境变量 ALIYUN_APPKEY)")
        sys.exit(1)
    
    # 0. 检查并获取 Token
    token = args.token
    if not token and not args.dry_run:
        # 尝试从参数获取，或者环境变量（可选支持）
        ak_id = args.ak_id or os.getenv('ALIYUN_AK_ID')
        ak_secret = args.ak_secret or os.getenv('ALIYUN_AK_SECRET')
        
        if ak_id and ak_secret:
            token = fetch_token(ak_id, ak_secret)
            if not token:
                logger.error("自动获取Token失败，无法继续")
                sys.exit(1)
        else:
            logger.error("未提供Token，且也未提供 AK/SK (参数或环境变量 ALIYUN_AK_ID/SECRET)，无法鉴权。")
            sys.exit(1)
    
    # 1. 准备目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 加载文本
    input_path = args.input_file
    if not os.path.exists(input_path) and input_path == AISHELL_FILE:
        candidates = [
            AISHELL_FILE,
            "aishell_transcript_v0.8.txt",
            os.path.join(os.path.dirname(__file__), AISHELL_FILE)
        ]
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break
                
    logger.info(f"加载文本: {input_path}")
    all_texts = load_texts(input_path)
    if not all_texts:
        logger.error("未找到有效的文本数据")
        return

    logger.info(f"总文本行数: {len(all_texts)}")
    
    # 3. 生成任务列表
    tasks = []
    
    # 遍历每种情感
    for emotion_key, config in EMOTION_CONFIG.items():
        selected_texts = random.sample(all_texts, min(len(all_texts), args.count))
        
        # 情感目录
        emotion_dir = os.path.join(args.output_dir, emotion_key)
        os.makedirs(os.path.join(emotion_dir, "wavs"), exist_ok=True)
        
        for i, text in enumerate(selected_texts):
            # 随机选择一个支持该情感的音色 (VOICE_POOL中的都支持这6种)
            voice = random.choice(VOICE_POOL)
            
            utt_id = f"{emotion_key}_{i:05d}"
            output_path = os.path.join(emotion_dir, "wavs", f"{utt_id}.mp3")
            
            tasks.append(GenTask(
                utt_id=utt_id,
                text=text,
                emotion_key=emotion_key,
                voice=voice,
                instruct=config['instruct'],
                output_path=output_path
            ))
            
    logger.info(f"计划生成 {len(tasks)} 个音频任务 (情感数: {len(EMOTION_CONFIG)}, 每种: {args.count})")
    
    # 4. 执行合成
    success_count = 0
    fail_count = 0
    
    if args.dry_run:
        logger.info("Dry Run模式: 跳过TTS合成步骤")
        # 伪造成功所有任务以便生成索引文件
        success_count = len(tasks)
        
        # 为了让索引文件生成逻辑跑通，我们需要确保wav文件"存在"。
        # 但在dry-run下我们不应该创建空文件污染环境，或者我们应该创建空文件？
        # 通常 dry-run 是为了验证逻辑。
        # 这里我们简单跳过合成，但为了生成索引文件，我们需要模拟 wav 文件已生成。
        # 或者我们修改索引生成逻辑，只检查 tasks 列表而不是磁盘文件。
        # 鉴于代码依赖 os.listdir(wav_dir)，dry-run 模式下如果不生成文件，后续索引生成会失效。
        # 让我们在 Tasks 循环中创建空的 dummy 文件 如果是 dry-run。
        
        for task in tasks:
            if not os.path.exists(task.output_path):
                # 创建空文件用于占位
                with open(task.output_path, 'wb') as f:
                    pass
        
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_task, task, args.appkey, token): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                task = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Worker exception: {e}")
                    fail_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"进度: {i + 1}/{len(tasks)} (成功: {success_count}, 失败: {fail_count})", end='\r')
                    
        print(f"\n合成完成! 成功: {success_count}, 失败: {fail_count}")
    
    # 5. 生成 Kaldi 索引文件
    for emotion_key, config in EMOTION_CONFIG.items():
        emotion_dir = os.path.join(args.output_dir, emotion_key)
        wav_dir = os.path.join(emotion_dir, "wavs")
        
        if not os.path.exists(wav_dir): continue
        
        files = sorted(os.listdir(wav_dir))
        wav_files = [f for f in files if f.endswith('.mp3')]
        
        if not wav_files: continue
        
        with open(os.path.join(emotion_dir, "wav.scp"), 'w', encoding='utf-8') as f_wav, \
             open(os.path.join(emotion_dir, "text"), 'w', encoding='utf-8') as f_text, \
             open(os.path.join(emotion_dir, "utt2spk"), 'w', encoding='utf-8') as f_utt2spk, \
             open(os.path.join(emotion_dir, "spk2utt"), 'w', encoding='utf-8') as f_spk2utt, \
             open(os.path.join(emotion_dir, "instruct.txt"), 'w', encoding='utf-8') as f_instruct:
            
            f_instruct.write(config['instruct']) 
            
            # 由于 speaker 现在是动态的，我们需要记录每个 utt 对应的 speaker
            # 重新建立映射
            emotion_tasks = [t for t in tasks if t.emotion_key == emotion_key]
            id_map = {t.utt_id: t for t in emotion_tasks}
            
            spk2utt_dict = {}
            
            for wav_file in wav_files:
                utt_id = os.path.splitext(wav_file)[0]
                task = id_map.get(utt_id)
                if not task: continue
                
                abs_path = os.path.abspath(os.path.join(wav_dir, wav_file))
                
                # 区分 Speaker ID: aliyun_voiceName (如 aliyun_zhimiao_emo)
                spk_id = f"aliyun_{task.voice}"
                
                # wav.scp
                f_wav.write(f"{utt_id} {abs_path}\n")
                # text
                f_text.write(f"{utt_id} {task.text}\n")
                # utt2spk
                f_utt2spk.write(f"{utt_id} {spk_id}\n")
                
                if spk_id not in spk2utt_dict:
                    spk2utt_dict[spk_id] = []
                spk2utt_dict[spk_id].append(utt_id)
            
            # spk2utt
            for spk, utts in spk2utt_dict.items():
                f_spk2utt.write(f"{spk} {' '.join(utts)}\n")

        logger.info(f"[{emotion_key}] 索引文件已生成 => {emotion_dir}")

    # 6. 后处理提示
    logger.info("="*60)
    logger.info("后续处理建议:")
    logger.info("1. 音频格式转换 (MP3 -> WAV 16k):")
    logger.info(f"   find {args.output_dir} -name '*.mp3' | while read f; do ffmpeg -y -i \"$f\" -ar 16000 -ac 1 \"${{f%.*}}.wav\"; done")
    logger.info("2. 运行 CosyVoice 数据预处理工具 (extract_embedding, extract_speech_token, make_parquet_list)")
    logger.info("="*60)

if __name__ == "__main__":
    main()
