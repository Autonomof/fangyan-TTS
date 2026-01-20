#!/usr/bin/env python3
"""
CosyVoice 微调前后推理对比脚本

用法:
    python compare_inference.py --output_dir ./comparison_output

功能:
    1. 加载原始预训练模型
    2. 加载微调后的模型
    3. 使用相同的文本和指令生成音频
    4. 保存对比结果到指定目录
"""

import os
import sys
import argparse
import shutil
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
    print(f"  生成: {text[:30]}...")
    
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


def main():
    parser = argparse.ArgumentParser(description="CosyVoice 微调前后推理对比")
    parser.add_argument("--pretrained_dir", type=str, 
                        default="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
                        help="预训练模型目录")
    parser.add_argument("--finetuned_llm", type=str,
                        default="CosyVoice/examples/dialect/cosyvoice3/exp/dialect/llm/torch_ddp/llm.pt",
                        help="微调后的 LLM 权重路径")
    parser.add_argument("--prompt_wav", type=str,
                        default="/sharedata/user/qianbin/CosyVoice/CosyVoice_newest/CosyVoice/person/person2.mp3",
                        help="参考音频路径")
    parser.add_argument("--output_dir", type=str,
                        default="./comparison_output",
                        help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 河南话文本
    henan_texts = [
        "马年得劲，愿恁家旺滴很！",
        "祝您马奔前程，好事不断捻儿！",
        "新年到，马儿扬鬃，福寿康宁嘞！",
        "马儿撒欢，新年福气哗啦啦来！",
        "愿年景暄腾腾，光景暖烘烘！",
        "愿你新年可得劲，百事都如意！",
        "新年马驹儿尥蹶子，甩来满院福！",
        "新年愿恁日子，跟马跑嘞样畅快！",
        "愿恁嘞喜气，跟马鬃样扬起来！",
        "祝恁龙马精神，身板儿硬梆梆！",
    ]

    # 湖南话文本
    hunan_texts = [
        "马年到，策马飙起，路路通泰！",
        "祝您马年行大运，福气漾得满堂红！",
        "马到成功，祝您事业蹬蹬滚！",
        "做事熨熨帖，愿新年顺遂无磕绊！",
        "祝你马年冲劲足，事业搞得嬲塞！",
        "马到成功，祝您新年事事熨帖！",
        "新年骏马驮福，祝你喜气满堂飘！",
        "愿新年劲鼓鼓，拼出个红火天道！",
        "祝您马年劲鼓鼓，事业霸得蛮！",
        "新年福星照，愿你日子韵味得咧！",
    ]
    
    # 方言配置：只测试河南话和湖南话
    dialects = [
        {"name": "henan", "instruct": "请用河南话说这句话。<|endofprompt|>", "texts": henan_texts},
        {"name": "hunan", "instruct": "请用湖南话说这句话。<|endofprompt|>", "texts": hunan_texts},
    ]
    
    # 生成测试用例：每个方言 x 每条文本
    test_cases = []
    for dialect in dialects:
        for i, text in enumerate(dialect["texts"], 1):
            test_cases.append({
                "name": dialect["name"],
                "text": text,
                "instruct": dialect["instruct"],
                "index": i
            })
    
    # 检查文件是否存在
    if not os.path.exists(args.pretrained_dir):
        print(f"错误: 预训练模型目录不存在: {args.pretrained_dir}")
        return
    
    if not os.path.exists(args.prompt_wav):
        print(f"错误: 参考音频不存在: {args.prompt_wav}")
        return
    
    print("=" * 60)
    print("CosyVoice 微调前后推理对比")
    print("=" * 60)
    
    # 1. 使用原始模型生成
    print("\n[1/2] 加载原始预训练模型...")
    original_model = load_model(args.pretrained_dir)
    
    print("\n生成原始模型音频...")
    for case in test_cases:
        output_path = output_dir / f"{case['name']}_{case['index']}_original.wav"
        generate_audio(
            original_model, 
            case['text'], 
            case['instruct'],
            args.prompt_wav,
            str(output_path)
        )
    
    # 释放原始模型显存
    del original_model
    torch.cuda.empty_cache()
    
    # 2. 使用微调后的模型生成
    if os.path.exists(args.finetuned_llm):
        print("\n[2/2] 加载微调后模型...")
        finetuned_model = load_model(args.pretrained_dir, args.finetuned_llm)
        
        print("\n生成微调后模型音频...")
        for case in test_cases:
            output_path = output_dir / f"{case['name']}_{case['index']}_finetuned.wav"
            generate_audio(
                finetuned_model,
                case['text'],
                case['instruct'],
                args.prompt_wav,
                str(output_path)
            )
        
        del finetuned_model
        torch.cuda.empty_cache()
    else:
        print(f"\n跳过微调模型: 权重文件不存在 {args.finetuned_llm}")
        print("训练完成后重新运行此脚本")
    
    print("\n" + "=" * 60)
    print(f"对比结果已保存到: {output_dir}")
    print("=" * 60)
    
    # 列出生成的文件
    print("\n生成的音频文件:")
    for f in sorted(output_dir.glob("*.wav")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
