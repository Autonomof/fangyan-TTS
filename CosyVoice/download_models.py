import os
from huggingface_hub import snapshot_download

# 1. 设置镜像站地址 (国内最常用的镜像站)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 定义模型列表和对应的存放路径
models = {
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512': 'pretrained_models/Fun-CosyVoice3-0.5B',
    'FunAudioLLM/CosyVoice2-0.5B': 'pretrained_models/CosyVoice2-0.5B',
}

# 3. 循环下载
for repo_id, local_path in models.items():
    print(f"正在从镜像站下载: {repo_id} ...")
    snapshot_download(
        repo_id=repo_id, 
        local_dir=local_path,
        # 建议加上 resume_download=True，防止大文件下载中断
        resume_download=True
    )

print("所有模型下载完成！")