#!/bin/bash
# 设置符号链接脚本
# 在 Linux 服务器上运行此脚本来创建正确的符号链接

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "创建符号链接..."

# 删除 Windows 下创建的文件（如果是文件而不是链接）
rm -f cosyvoice tools 2>/dev/null

# 创建符号链接
ln -sf ../../../cosyvoice cosyvoice
ln -sf ../../../tools tools

echo "符号链接创建完成:"
ls -la cosyvoice tools

echo ""
echo "验证 PYTHONPATH..."
source ./path.sh
python -c "from cosyvoice.dataset.dataset import Dataset; print('✓ cosyvoice.dataset OK')"
python -c "from matcha.utils.audio import mel_spectrogram; print('✓ matcha OK')"

echo ""
echo "设置完成！现在可以运行: bash run.sh"
