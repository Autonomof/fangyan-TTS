#!/usr/bin/env python3
# coding=utf-8
"""
方言推理对比测试脚本 (多方言单文本版)

用法:
    python dialect_inference_test.py --output_dir ./dialect_test_output

功能:
    1. 使用固定文本: '祝您新年，龙马精神，骏业腾飞'
    2. 支持多种方言指令对比
    3. 支持原始模型与微调模型对比
"""

import os
import sys
import argparse
from pathlib import Path

# 添加 CosyVoice 路径
# 确保在当前目录下或者根据实际结构调整
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio

def load_model(model_dir, finetuned_llm_path=None):
    """加载模型，可选择替换微调后的 LLM 权重"""
    from cosyvoice.cli.cosyvoice import AutoModel
    
    print(f"正在加载模型: {model_dir}")
    model = AutoModel(model_dir=model_dir)
    
    if finetuned_llm_path and os.path.exists(finetuned_llm_path):
        print(f"正在应用微调权重: {finetuned_llm_path}")
        state_dict = torch.load(finetuned_llm_path, map_location='cpu')
        # 过滤掉非模型参数
        model_state = {k: v for k, v in state_dict.items() if not k.startswith('epoch') and not k.startswith('step')}
        model.model.llm.load_state_dict(model_state, strict=False)
        print("微调权重加载完成")
    
    return model

def generate_audio(model, text, instruct, prompt_wav, output_path):
    """合成音频"""
    print(f"  合成指令: '{instruct}' -> {output_path}")
    
    # 使用 inference_instruct2
    try:
        for result in model.inference_instruct2(text, instruct, prompt_wav):
            audio = result['tts_speech']
            torchaudio.save(output_path, audio, model.sample_rate)
            break
    except Exception as e:
        print(f"  合成失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="CosyVoice 多方言推理对比测试")
    parser.add_argument("--pretrained_dir", type=str, 
                        default="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
                        help="预训练模型目录")
    parser.add_argument("--finetuned_llm", type=str,
                        default="CosyVoice/examples/dialect/cosyvoice3/exp/dialect/llm/torch_ddp/llm.pt",
                        help="微调后的 LLM 权重路径 (可选)")
    parser.add_argument("--prompt_wav", type=str,
                        default="/sharedata/user/qianbin/xiaowu.wav",
                        help="参考音频路径")
    parser.add_argument("--output_dir", type=str,
                        default="./dialect_test_output",
                        help="音频输出目录")
    parser.add_argument("--text", type=str,
                        default="祝您新年，龙马精神，骏业腾飞",
                        help="待合成的文本内容趋势")
    
    args = parser.parse_args()
    
    # 路径检查
    if not os.path.exists(args.pretrained_dir):
        print(f"致命错误: 找不到模型目录 {args.pretrained_dir}")
        return
    
    if not os.path.exists(args.prompt_wav):
        print(f"警告: 参考音频 {args.prompt_wav} 不存在，请确保路径正确或通过 --prompt_wav 指定。")
        # 如果是开发环境测试，可以容忍，但实际运行会报错

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 方言配置
    # 格式: { key: { "display": 显示名称, "instruct": 指令文本 } }
    dialects = {
        # 原有 compare_inference.py 中的方言
        "hunan": "湖南话",
        "henan": "河南话",
        "cantonese": "粤语",
        "tianjin": "天津话",
        "sichuan": "四川话",
        "zhengzhou": "郑州话",
        "hunan_pu": "湖南普通话",
        "dongbei": "东北话",
        "xian": "西安话",
        "shanghai": "上海话",
        "guangxi": "广西话",
        # 额外新增的方言
        "minnan": "闽南语",
        "gansu": "甘肃话",
        "ningxia": "宁夏话",
        # 普通话 (空指令)
        "mandarin": None
    }

    # 1. 准备模型 (检测是否需要加载两个模型做对比)
    run_comparison = os.path.exists(args.finetuned_llm) if args.finetuned_llm else False
    
    print("=" * 60)
    print(f"文本: {args.text}")
    print(f"参考音频: {args.prompt_wav}")
    print("=" * 60)

    # 运行原始模型
    print("\n[阶段 1] 运行原始预训练模型...")
    original_model = load_model(args.pretrained_dir)
    
    for key, display in dialects.items():
        instruct = f"请用{display}说。<|endofprompt|>" if display else ""
        out_fn = f"{key}_original.wav"
        generate_audio(original_model, args.text, instruct, args.prompt_wav, str(output_root / out_fn))
    
    del original_model
    torch.cuda.empty_cache()

    # 如果有微调权重，运行微调模型
    if run_comparison:
        print("\n[阶段 2] 运行微调后的模型...")
        finetuned_model = load_model(args.pretrained_dir, args.finetuned_llm)
        
        for key, display in dialects.items():
            instruct = f"请用{display}说。<|endofprompt|>" if display else ""
            out_fn = f"{key}_finetuned.wav"
            generate_audio(finetuned_model, args.text, instruct, args.prompt_wav, str(output_root / out_fn))
        
        del finetuned_model
        torch.cuda.empty_cache()
    else:
        print("\n[跳过] 未发现微调权重，仅生成原始模型音频。")

    print("\n" + "=" * 60)
    print(f"测试完成！文件保存至: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
