#!/bin/bash
# Path configuration for dialect training

# 获取 CosyVoice 根目录 (从 examples/dialect 向上两级)
COSYVOICE_ROOT=$(dirname $(dirname $(dirname $(realpath "$0"))))

# 添加 CosyVoice 根目录到 PYTHONPATH
export PYTHONPATH="${COSYVOICE_ROOT}:${PYTHONPATH}"

# 添加 third_party/Matcha-TTS 到 PYTHONPATH (某些模块需要)
if [ -d "${COSYVOICE_ROOT}/third_party/Matcha-TTS" ]; then
    export PYTHONPATH="${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH}"
fi

echo "PYTHONPATH set to: ${PYTHONPATH}"
