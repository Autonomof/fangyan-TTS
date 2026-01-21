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
    
    # 各方言文本（每种方言 10 句，按各自的说法）
    dialect_texts = {
        "hunan": [
            "满哥，这一向在搞墨子咯？",
            "妹佗，长得蛮标志咧。",
            "今天天气真的好韵味，适合出去以此。",
            "你恰饭了冒咯？",
            "搞醉哒，这个细伢子真是不听话。",
            "快点啰，别的都在等你示。",
            "这碗米粉的味道硬是索，辣得过瘾。",
            "你有蛮灵泛咧，这道题一下子就做出来哒。",
            "莫在那里宝里宝气哒，看着都着急。",
            "我是湖南人，恰得苦，耐得烦，霸得蛮。",
        ],
        "henan": [
            "你那是弄啥嘞？",
            "这几天热哩很，都不想出门。",
            "今儿个黑了，明天再说吧。",
            "你给我站那儿，别动！",
            "你看那个人，长哩真排场。",
            "这碗面真好吃，得劲！",
            "夜儿个晚上，你干啥去啦？",
            "你别给我胡吊扯，我才不信嘞。",
            "这事儿你看咋办？中不中？",
            "乖乖，这楼盖哩真高！",
        ],
        "cantonese": [
            "你好吗",
            "早晨啊",
            "食咗饭未啊",
            "我去街市买菜",
            "今晚食咩好呢",
            "唔好意思借过",
            "多谢晒",
            "麻烦晒你",
            "唔该借借",
            "听日得闲吗",
        ],
        "tianjin": [
            "吃了吗您内",
            "早点吃的嘛",
            "两根馃子一碗浆子",
            "这天儿真冷啊",
            "多穿点别冻着",
            "您去哪啊",
            "我去上班儿",
            "这道儿怎么走啊",
            "直走过了红绿灯就到了",
            "谢谢您了",
        ],
        "sichuan": [
            "你切哪点",
            "我在解放碑等你",
            "要得嘛",
            "好久不见",
            "你瘦了好多",
            "这就是生活",
            "吃火锅不",
            "多放点海椒",
            "微辣就是妥协",
            "我要特辣",
        ],
        "zhengzhou": [
            "中不中",
            "那肯定中啊",
            "吃了吗",
            "刚吃过",
            "吃的啥",
            "吃的烩面",
            "那家烩面可得劲儿了",
            "大碗宽面",
            "羊肉汤",
            "漂着香菜",
        ],
        "hunan_pu": [
            "你好咯",
            "最近好不咯",
            "恰饭哒冒",
            "我恰饱哒",
            "你克哪凯咯",
            "我克逛街",
            "一起克不咯",
            "好哒一起克",
            "你搞么子鬼咯",
            "莫港样咯",
        ],
        "dongbei": [
            "哎呀妈呀，这天儿咋这么冷呢，冻得我直打哆嗦。",
            "你瞅啥？瞅你咋地！再瞅一个试试？",
            "干哈呢你？一天天五脊六兽的，也没个正事儿。",
            "别跟我扯犊子，我才不信你那套鬼话呢。",
            "麻溜利索的，别在那儿磨磨唧唧的，看着都着急。",
            "这事儿你得整明白喽，别到时候整得秃噜反帐的。",
            "哎呦我去，这小烧烤味儿，贼拉香，太带劲了。",
            "你别在那儿得瑟了，小心出门卡个跟头。",
            "听我说，这玩意儿备不住以后能用上，先留着吧。",
            "今儿个真高兴，咱哥俩必须得喝点儿，不醉不归。",
        ],
        "xian": [
            "额滴神啊，这事儿咋弄成这模样咧？",
            "这是弄啥嘛？一天天把人愁滴。",
            "走，伙计，请你咥泡馍去，把人美滴很！",
            "这面条筋道得很，油泼辣子一道，简直嘹咋咧！",
            "你个瓜怂，咋这么不听话呢？非得让人操心。",
            "么麻达，这事儿包在额身上，绝对给你办好。",
            "快点，克里马擦滴，别磨磨蹭蹭像个大姑娘。",
            "你看那个碎怂，一天光知道玩，也不学习。",
            "这天气热得让人受不了，赶紧回屋歇着去。",
            "你怎么这么瓷马二楞滴？做事不动动脑子。",
        ],
        "shanghai": [
            "侬好，今朝天气蛮好个，出去兜兜风伐？",
            "谢谢侬哦，侬真是个好人，帮了我大忙了。",
            "帮帮忙好伐？这种事情侬也做得出来，真是拎不清。",
            "侬脑子瓦特啦？格能简单个道理都唔没想明白。",
            "今朝个菜味道蛮灵个，特特是迭个红烧肉，嗲！",
            "侬勿要勒海捣浆糊了，有啥讲啥，爽气点。",
            "伊个人老结棍个，一口气跑了十公里，面不改色。",
            "侬看伊那个样子，神气活现，像煞有介事一样。",
            "大家出来白相，最重要是开心，勿要计较格能多。",
            "侬额骨头碰着天花板了，运气哪能格能好？",
        ],
        "guangxi": [
            "今天怎么这种天气，热得我想死。",
            "喂，友仔，今晚出来吃烧烤啵？",
            "你这个人怎么这么颠，我都懒得理你。",
            "这碗粉太好吃了，我要连汤都喝完。",
            "哎呀，我不小心摔了一跤，膝盖都磕破了。",
            "你别老是讲这种话，听得我心里蓝瘦香菇。",
            "那个女的好漂亮捏，是不是你女朋友啊？",
            "快点走啦，磨磨蹭蹭的，等下赶不上车了。",
            "我也想去旅游，可是没钱，太惨了。",
            "你不要骗我哦，我这个人很单纯的。",
        ],
        # 普通话：instruct 为空
        "mandarin": [
            "你在干什么？",
            "这几天太热了，都不想出门。",
            "今天太晚了，明天再说吧。",
            "你站在那儿别动！",
            "你看那个人，长得真精神。",
            "这碗面真好吃，特别劲道！",
            "昨晚你去哪儿了？",
            "别跟我胡说，我才不信。",
            "这事你看怎么办，行不行？",
            "哎呀，这楼盖得真高！",
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
