# fangyan-TTS

🗣️ **方言 TTS 数据集生成工具** - 用于生成湖南话和河南话的 TTS 训练数据，供 CosyVoice3 微调使用。

## 项目简介

本项目用于生成方言语音数据集，通过调用火山引擎 TTS API 合成带有方言口音的语音数据，用于微调 CosyVoice3 模型，提升其湖南话和河南话的合成能力。

### 数据策略

| 方言 | 数据来源 | 音色 | 样本数 |
|------|----------|------|--------|
| 湖南话 | AIShell 前1000条 + hunan.txt | 长沙靓女 (BV216_streaming) | ~1191 |
| 河南话 | AIShell 后1000条 + henan.txt | 乡村企业家 (BV214_streaming) | ~1190 |

## 快速开始

### 1. 安装依赖

```bash
pip install requests python-dotenv
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

### 3. 准备数据

确保以下文件存在：
- `aishell_transcript_v0.8.txt` - AIShell 数据集转录文本
- `hunan.txt` - 湖南话方言语料
- `henan.txt` - 河南话方言语料

### 4. 生成数据集

```bash
# 仅生成索引文件（不调用 API）
python generate_dialect_dataset.py --dry-run

# 生成全部数据
python generate_dialect_dataset.py --mode all

# 仅生成湖南话 / 河南话
python generate_dialect_dataset.py --mode hunan
python generate_dialect_dataset.py --mode henan

# 调整 API 请求频率
python generate_dialect_dataset.py --mode all --qps 5
```

## 输出格式

生成的数据集采用 Kaldi 格式，兼容 CosyVoice3 训练：

```
dataset/
├── hunan/
│   ├── wav.scp      # 音频路径索引
│   ├── text         # 文本标注
│   ├── utt2spk      # 语音到说话人映射
│   └── spk2utt      # 说话人到语音映射
└── henan/
    ├── wav.scp
    ├── text
    ├── utt2spk
    └── spk2utt
```

## CosyVoice3 微调流程

1. **转换音频格式** (MP3 → WAV 16kHz mono)
2. **提取 Speaker Embedding**
3. **提取 Speech Token**
4. **生成 Parquet 格式**
5. **开始训练**

详细步骤参考 `CosyVoice/examples/libritts/cosyvoice3/run.sh`

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_dialect_dataset.py` | 主脚本：批量 TTS 合成 + 索引生成 |
| `hunan.txt` | 湖南话方言语料（~191句） |
| `henan.txt` | 河南话方言语料（~190句） |
| `doubao_tts.py` | 原始 TTS Demo（参考用） |
| `.env.example` | 环境变量配置模板 |

## 依赖

- Python 3.8+
- requests
- python-dotenv (可选)

## 许可证

MIT License

## 致谢

- [火山引擎 TTS](https://www.volcengine.com/product/tts) - 语音合成 API
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 语音合成模型
- [AIShell](http://www.openslr.org/33/) - 中文语音数据集
