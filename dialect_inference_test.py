#!/usr/bin/env python3
"""
特定方言TTS推理生成脚本

用法:
    python dialect_inference_test.py --output_dir ./dialect_inference_results

功能:
    使用指定文本和参考音频，生成闽南话、甘肃话、宁夏话、普通话的TTS音频。
    文本: '祝您新年，龙马精神，骏业腾飞'
"""

import os
import sys
import argparse
from pathlib import Path

# 添加 CosyVoice 路径
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio

def load_model(model_dir, finetuned_llm_path=None):
    """加载模型，可选择替换微调后的 LLM 权重"""
    from cosyvoice.cli.cosyvoice import AutoModel
    
    print(f"加载模型: {model_dir}")
    model = AutoModel(model_dir=model_dir)
    
    if finetuned_llm_path and os.path.exists(finetuned_llm_path):
        print(f"替换 LLM 权重: {finetuned_llm_path}")
        # 加载微调后的 LLM 权重
        state_dict = torch.load(finetuned_llm_path, map_location='cpu')
        # 移除 'epoch' 和 'step' 等非模型参数
        model_state = {k: v for k, v in state_dict.items() if not k.startswith('epoch') and not k.startswith('step')}
        model.model.llm.load_state_dict(model_state, strict=False)
        print("LLM 权重替换完成")
    
    return model

def generate_audio(model, text, instruct, prompt_wav, output_path, stream=False):
    """生成音频并保存"""
    print(f"  生成: {text} | 指令: {instruct}")
    
    try:
        for i, result in enumerate(model.inference_instruct2(
            text, 
            instruct, 
            prompt_wav, 
            stream=stream
        )):
            audio = result['tts_speech']
            torchaudio.save(output_path, audio, model.sample_rate)
            print(f"  保存到: {output_path}")
            break  # 只保存第一个结果
    except Exception as e:
        print(f"  生成失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="特定方言TTS推理生成")
    parser.add_argument("--pretrained_dir", type=str, 
                        default="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
                        help="预训练模型目录")
    parser.add_argument("--finetuned_llm", type=str,
                        default="CosyVoice/examples/dialect/cosyvoice3/exp/dialect/llm/torch_ddp/llm.pt",
                        help="微调后的 LLM 权重路径 (可选，存在则加载)")
    parser.add_argument("--prompt_wav", type=str,
                        default="/sharedata/user/qianbin/xiaowu.wav",
                        help="参考音频路径")
    parser.add_argument("--output_dir", type=str,
                        default="./dialect_inference_results",
                        help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 统一文本
    target_text = "祝您新年，龙马精神，骏业腾飞"
    
    # 方言配置
    # 闽南话、甘肃话、宁夏话、普通话
    dialects = [
        {
            "name": "minnan",
            "desc": "闽南话",
            "instruct": "请用闽南话说<|endofprompt|>"
        },
        {
            "name": "gansu",
            "desc": "甘肃话",
            "instruct": "请用甘肃话说<|endofprompt|>"
        },
        {
            "name": "ningxia",
            "desc": "宁夏话",
            "instruct": "请用宁夏话说<|endofprompt|>"
        },
        {
            "name": "mandarin",
            "desc": "普通话",
            "instruct": "<|endofprompt|>" # 普通话instruct为空
        }
    ]
    
    # 检查文件是否存在
    if not os.path.exists(args.pretrained_dir):
        print(f"错误: 预训练模型目录不存在: {args.pretrained_dir}")
        return
    
    # 如果默认参考音频不存在，检查是否能用之前的替代
    if not os.path.exists(args.prompt_wav):
        print(f"警告: 参考音频 {args.prompt_wav} 不存在。")
        # 尝试使用 compare_inference.py 中的默认路径作为备选
        fallback_wav = "/sharedata/user/qianbin/CosyVoice/CosyVoice_newest/CosyVoice/person/person2.mp3"
        if os.path.exists(fallback_wav):
            print(f"使用备选音频: {fallback_wav}")
            args.prompt_wav = fallback_wav
        else:
            print("错误: 未找到可用的参考音频，请使用 --prompt_wav 指定。")
            return
    
    print("=" * 60)
    print("特定方言TTS推理生成")
    print(f"参考音频: {args.prompt_wav}")
    print(f"文本: {target_text}")
    print("=" * 60)
    
    # 加载模型
    # 如果有微调权重，优先加载微调后模型
    load_finetuned = False
    if os.path.exists(args.finetuned_llm):
        print(f"\n检测到微调权重: {args.finetuned_llm}")
        print("将加载【微调后】模型进行推理...")
        model = load_model(args.pretrained_dir, args.finetuned_llm)
        load_finetuned = True
    else:
        print("\n未检测到微调权重，将加载【原始预训练】模型...")
        model = load_model(args.pretrained_dir)
    
    print("\n开始生成音频...")
    
    for dialect in dialects:
        file_suffix = "finetuned" if load_finetuned else "original"
        output_filename = f"{dialect['name']}_{file_suffix}.wav"
        output_path = output_dir / output_filename
        
        print(f"\n正在生成 [{dialect['desc']}] ...")
        generate_audio(
            model,
            target_text,
            dialect['instruct'],
            args.prompt_wav,
            str(output_path)
        )
        
    print("\n" + "=" * 60)
    print(f"生成完成！结果保存到: {output_dir}")
    print("=" * 60)
    
    # 列出生成的文件
    print("\n生成的音频文件:")
    for f in sorted(output_dir.glob("*.wav")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
