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
    
    # 各方言文本（每种方言 10 句，同一序号表达同一含义）
    dialect_texts = {
        "hunan": [
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
        ],
        "henan": [
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
        ],
        "cantonese": [
            "马年好犀利，愿你屋企兴旺发达！",
            "祝你前程似锦，好事不断嚟！",
            "新年到喇，祝你福寿安康！",
            "马仔撒欢，新年福气滚滚嚟！",
            "愿新年红红火火，日子暖笠笠！",
            "祝你新年顺顺利利，事事如意！",
            "新年骏马起跳，满院福气到！",
            "愿你新年日子好似骏马飞奔咁畅快！",
            "愿你嘅喜气似马鬃一样飞扬！",
            "祝你龙马精神，身体硬净！",
        ],
        "tianjin": [
            "马年可得劲儿，愿您家旺滴很！",
            "祝您前程倍儿好，好事不断！",
            "新年到喽，祝您福寿安康！",
            "马儿撒欢儿，新年福气哗啦啦来！",
            "愿新年红火，日子暖烘烘！",
            "祝您新年顺溜儿，事事如意！",
            "新年骏马一蹿，满院福气到！",
            "愿您新年日子跟马跑似的畅快！",
            "愿您的喜气跟马鬃一样扬起来！",
            "祝您龙马精神，身板儿硬朗！",
        ],
        "sichuan": [
            "马年巴适得很，愿你屋头兴旺！",
            "祝你前程要得，好事不断哈！",
            "新年到了，祝你福寿安康哈！",
            "马儿撒欢，新年福气滚滚来哈！",
            "愿新年红火，日子暖和和！",
            "祝你新年顺顺当当，事事如意哈！",
            "新年骏马一蹦，满院福气来！",
            "愿你新年日子像骏马跑起一样畅快！",
            "愿你嘞喜气像马鬃一样飘起！",
            "祝你龙马精神，身体硬是好！",
        ],
        "zhengzhou": [
            "马年得劲儿，愿恁家旺滴很！",
            "祝您马奔前程，好事不断嘞！",
            "新年到，马儿扬鬃，福寿安康嘞！",
            "马儿撒欢，新年福气哗啦啦来！",
            "愿年景红火，光景暖烘烘！",
            "愿你新年可得劲，百事都如意！",
            "新年马驹儿一蹦，满院福气来！",
            "新年愿恁日子，跟马跑样畅快！",
            "愿恁嘞喜气，跟马鬃样扬起来！",
            "祝恁龙马精神，身板儿硬朗！",
        ],
        "hunan_pu": [
            "马年蛮得劲咯，愿你家里旺哒！",
            "祝你前程好咯，好事不断哒！",
            "新年到咯，祝你福寿安康！",
            "马儿撒欢咯，新年福气哗啦啦来哒！",
            "愿新年红火咯，日子暖烘烘！",
            "祝你新年顺顺哒，事事如意咯！",
            "新年骏马一蹦，满院福气来咯！",
            "愿你新年日子像马儿跑咯样畅快！",
            "愿你嘞喜气像马鬃一样扬起来咯！",
            "祝你龙马精神，身板硬朗咯！",
        ],
        "dongbei": [
            "马年贼拉得劲，愿你家里旺得很！",
            "祝你前程倍儿好，好事不断！",
            "新年到了，祝你福寿安康！",
            "马儿撒欢，新年福气哗啦啦来！",
            "愿新年红火，日子暖烘烘！",
            "祝你新年顺顺当当，事事如意！",
            "新年骏马一蹿，满院福气到！",
            "愿你新年日子跟马跑似的贼拉畅快！",
            "愿你喜气跟马鬃一样飞扬！",
            "祝你龙马精神，身板儿倍儿硬朗！",
        ],
        "xian": [
            "马年好得很咧，愿你屋里旺滴很！",
            "祝你前程好得很，好事不断来咧！",
            "新年到咧，祝你福寿安康咧！",
            "马儿撒欢咧，新年福气哗啦啦来咧！",
            "愿新年红火，日子暖烘烘咧！",
            "祝你新年顺顺当当，事事如意咧！",
            "新年骏马一蹦，满院福气到咧！",
            "愿你新年日子像马跑一样畅快咧！",
            "愿你嘞喜气像马鬃一样扬起来咧！",
            "祝你龙马精神，身板硬朗咧！",
        ],
        "shanghai": [
            "马年蛮来赛，愿侬屋里兴旺发达伐！",
            "祝侬前程蛮好，好事不断来！",
            "新年到哉，祝侬福寿安康！",
            "马儿撒欢，新年福气滚滚来伐！",
            "愿新年红火，日脚暖笃笃！",
            "祝侬新年顺顺当当，事事如意！",
            "新年骏马一蹦，满院福气来！",
            "愿侬新年日脚像马跑一样畅快！",
            "愿侬嘞喜气像马鬃一样扬起来！",
            "祝侬龙马精神，身子骨硬朗！",
        ],
        "guangxi": [
            "马年好得很咯，愿你屋头旺哒！",
            "祝你前程好咯，好事不断来咯！",
            "新年到咯，祝你福寿安康！",
            "马儿撒欢咯，新年福气哗啦啦来！",
            "愿新年红火，日子暖烘烘咯！",
            "祝你新年顺顺咯，事事如意咯！",
            "新年骏马一蹦，满院福气来咯！",
            "愿你新年日子像马跑咯样畅快！",
            "愿你嘞喜气像马鬃一样扬起来咯！",
            "祝你龙马精神，身板硬朗咯！",
        ],
        # 普通话：instruct 为空
        "mandarin": [
            "马年真给力，愿你家兴旺发达！",
            "祝你前程似锦，好事不断！",
            "新年到了，祝你福寿安康！",
            "马儿撒欢，新年福气滚滚来！",
            "愿新年红火，日子暖融融！",
            "祝你新年顺顺当当，事事如意！",
            "新年骏马起跳，满院福气来！",
            "愿你新年日子像骏马奔跑般畅快！",
            "愿你的喜气像马鬃一样飞扬！",
            "祝你龙马精神，身体硬朗！",
        ],
    }

    dialect_display_names = {
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
    }

    # 方言配置：覆盖所有方言 + 普通话（普通话 instruct 为空）
    dialects = []
    for name, texts in dialect_texts.items():
        if name == "mandarin":
            instruct = ""
        else:
            display = dialect_display_names.get(name, name)
            instruct = f"请用{display}说。<|endofprompt|>"
        dialects.append({"name": name, "instruct": instruct, "texts": texts})
    
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
