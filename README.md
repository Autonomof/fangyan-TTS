# fangyan-TTS

🗣️ **方言 & 情感 TTS 数据集生成工具** - 用于生成方言和情感语音的 TTS 训练数据，供 CosyVoice3 微调使用。

## 项目简介

本项目用于生成方言和情感语音数据集，支持：
- 通过火山引擎 TTS API 合成多种方言语音
- 处理 ESD 情感语音数据集
- 微调 CosyVoice3 模型，提升其方言和情感合成能力

### 支持的方言

| 方言 | 音色 | 标识 |
|------|------|------|
| 湖南话 | 长沙靓女 (BV216) | `hunan` |
| 河南话 | 乡村企业家 (BV214) | `henan` |
| 四川话 | 川妹儿 (BV215) | `sichuan` |
| 东北话 | 东北老铁 (BV021) | `dongbei` |
| 天津话 | 天津哥 (BV212) | `tianjin` |
| 粤语 | 粤语女声 (BV218) | `cantonese` |
| 上海话 | 上海阿姨 (BV217) | `shanghai` |
| 西安话 | 西安老陕 (BV210) | `xian` |
| 广西话 | 广西老表 (BV213) | `guangxi` |

## 快速开始

### 1. 安装依赖

```bash
pip install requests python-dotenv mutagen tqdm torch torchaudio onnxruntime
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入火山引擎 API 凭据：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
VOLCENGINE_APPID=your_appid
VOLCENGINE_ACCESS_TOKEN=your_access_token
VOLCENGINE_CLUSTER=volcano_tts
```

## 工具脚本

### 数据生成脚本

| 脚本 | 说明 | 用法 |
|------|------|------|
| `generate_dialect_dataset.py` | 批量 TTS 合成方言数据 | 需要火山引擎 API |
| `generate_esd_dataset.py` | 处理 ESD 情感数据集 | 无需 API |
| `generate_emotion_dataset.py` | 情感数据集生成（FunASR 版） | 需要 ASR |

### 训练准备脚本

| 脚本 | 说明 |
|------|------|
| `prepare_training_data.py` | 训练数据准备（instruct生成、音频转换、合并） |
| `compare_inference.py` | 微调前后推理对比 |
| `stats_duration.py` | 统计各方言音频总时长 |

### 测试脚本

| 脚本 | 说明 |
|------|------|
| `doubao_tts.py` | 火山引擎 TTS API 测试脚本 |

---

## 详细用法

### 1. 生成方言 TTS 数据集 (`generate_dialect_dataset.py`)

使用火山引擎 TTS API 批量合成方言语音：

```bash
# 仅生成索引文件（不调用 API）
python generate_dialect_dataset.py --dry-run

# 生成全部方言数据
python generate_dialect_dataset.py --mode all

# 仅生成指定方言
python generate_dialect_dataset.py --mode hunan
python generate_dialect_dataset.py --mode henan

# 调整 API 请求频率
python generate_dialect_dataset.py --mode all --qps 5
```

**输出目录**: `dataset_new/<dialect>/`

---

### 2. 处理 ESD 情感数据集 (`generate_esd_dataset.py`)

处理 [ESD (Emotional Speech Dataset)](https://github.com/HLTSingapore/Emotional-Speech-Data) 数据集：

```bash
# 确保 ESD 数据集在 ./ESD 目录
python generate_esd_dataset.py
```

**情感分类**:
- `Happy` + `Surprise` → `请以开心高兴的语气用普通话说<|endofprompt|>`
- `Neutral` → `请以正常中立的语气用普通话说<|endofprompt|>`

**输出目录**: `dataset_emotion/`

---

### 3. 训练数据准备 (`prepare_training_data.py`)

合并方言和情感数据，生成训练所需的 Kaldi 格式文件：

```bash
# 完整准备流程（推荐）
python prepare_training_data.py --mode all --dataset-dir dataset_new

# 只生成 instruct 文件
python prepare_training_data.py --mode instruct

# 合并方言数据 + 情感数据
python prepare_training_data.py \
    --dataset-dir dataset_new \
    --extra-dirs dataset_emotion \
    --mode combine

# 指定输出目录
python prepare_training_data.py \
    --dataset-dir dataset_new \
    --extra-dirs dataset_emotion \
    --output-dir /path/to/output/combined \
    --mode combine

# 验证数据完整性
python prepare_training_data.py --mode validate
```

**参数说明**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式: all/instruct/combine/convert/validate | all |
| `--dataset-dir` | 主数据集目录 | dataset |
| `--extra-dirs` | 额外数据目录（如 dataset_emotion） | 无 |
| `--output-dir` | 合并输出目录（绝对路径） | dataset-dir/combined |
| `--force-instruct` | 强制重新生成 instruct | False |
| `--no-convert` | 跳过 MP3→WAV 转换 | False |

---

### 4. 微调前后推理对比 (`compare_inference.py`)

比较微调前后的模型效果：

```bash
python compare_inference.py     --pretrained_dir CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B     --finetuned_llm CosyVoice/examples/dialect/cosyvoice3/exp/dialect_emotion/llm/torch_ddp/epoch_20_whole.pt     --prompt_wav /sharedata/user/qianbin/yanlaodeng.wav     --output_dir ./comparison_output_emo-emo_instruct &
```

**输出**: 为每个方言和文本生成 `*_original.wav` 和 `*_finetuned.wav` 对比音频。

---

### 5. 统计音频时长 (`stats_duration.py`)

统计各方言数据集的音频总时长：

```bash
# 统计所有方言
python stats_duration.py

# 只统计指定方言
python stats_duration.py --dialect hunan

# JSON 格式输出
python stats_duration.py --format json

# 显示详细进度
python stats_duration.py --detailed
```

---

## CosyVoice3 微调训练

### 前置准备

1. **下载预训练模型**

   ```bash
   mkdir -p CosyVoice/pretrained_models
   cd CosyVoice/pretrained_models
   git lfs install
   git clone https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B Fun-CosyVoice3-0.5B
   ```

2. **安装 CosyVoice 依赖**

   ```bash
   cd CosyVoice
   pip install -r requirements.txt
   ```

### 训练步骤

> ⚠️ 推荐使用 **Linux 或 WSL2** 进行训练，需要 CUDA 支持。

```bash
cd CosyVoice/examples/dialect

# 按阶段执行（修改 run.sh 中的 stage 和 stop_stage）

# Stage 0: 生成 instruct 文件
# Stage 1: 提取 Speaker Embedding
# Stage 2: 提取 Speech Token (需要 GPU)
# Stage 3: 生成 Parquet 格式
# Stage 4: 合并数据列表
# Stage 5: 训练模型 (需要 GPU)

bash run.sh
```

### 详细流程说明

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 0 | `text` | `instruct` | 生成训练指令文件 |
| 1 | `wav.scp`, `utt2spk` | `utt2embedding.pt`, `spk2embedding.pt` | 提取说话人特征 |
| 2 | `wav.scp` | `utt2speech_token.pt` | 提取语音 Token |
| 3 | 上述所有 | `parquet/*.tar` | 打包为训练格式 |
| 5 | `data.list` | `exp/dialect/llm/` | 微调 LLM 模型 |

### 硬件要求

- **GPU**: 至少 16GB 显存 (推荐 24GB+)
- **内存**: 32GB+
- **磁盘**: 50GB+ (用于模型和数据)

---

## 输出格式

生成的数据集采用 Kaldi 格式，兼容 CosyVoice3 训练：

```
dataset_new/
├── hunan/
│   ├── wavs/        # 音频文件
│   ├── wav.scp      # 音频路径索引
│   ├── text         # 文本标注
│   ├── utt2spk      # 语音到说话人映射
│   ├── spk2utt      # 说话人到语音映射
│   └── instruct     # 训练指令
├── henan/
│   └── ...
├── combined/        # 合并后的数据
│   └── ...
└── dataset_cache.json

dataset_emotion/
├── wav.scp
├── text
├── utt2spk
├── spk2utt
└── instruct.txt     # 情感指令（逐句不同）
```

---

## 文件说明

| 文件/目录 | 说明 |
|------|------|
| `hunan.txt`, `henan.txt` 等 | 方言语料文本 |
| `aishell_transcript_v0.8.txt` | AIShell 转录文本 |
| `.env.example` | 环境变量配置模板 |
| `CosyVoice/` | CosyVoice3 模型源码（子模块） |
| `ESD/` | ESD 情感数据集（需自行下载） |

---

## 依赖

- Python 3.8+
- requests, python-dotenv, mutagen, tqdm
- torch, torchaudio (用于训练)
- onnxruntime (用于特征提取)
- funasr (用于 ASR，可选)

## 许可证

MIT License

## 致谢

- [火山引擎 TTS](https://www.volcengine.com/product/tts) - 语音合成 API
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 语音合成模型
- [AIShell](http://www.openslr.org/33/) - 中文语音数据集
- [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) - 情感语音数据集