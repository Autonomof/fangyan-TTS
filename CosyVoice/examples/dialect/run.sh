#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# Modified for dialect (Hunanese/Henanese) fine-tuning
#
# 使用方法:
#   1. 确保已下载预训练模型到 pretrained_models/Fun-CosyVoice3-0.5B
#   2. 确保已生成方言数据（运行 generate_dialect_dataset.py）
#   3. 确保已将 MP3 转换为 WAV（16kHz mono）
#   4. 按 stage 顺序执行训练流程
#
# Windows 用户注意:
#   - 建议使用 WSL2 或 Linux 服务器进行训练
#   - 本脚本在 Windows 下需要通过 Git Bash 或 WSL 运行

. ./path.sh || exit 1;

# ==================== 配置区 ====================

stage=0
stop_stage=5

# 数据目录 (相对于项目根目录)
data_dir=../../../dataset
# 预训练模型目录 (相对于 examples/dialect 目录)
pretrained_model_dir=../../pretrained_models/Fun-CosyVoice3-0.5B

# 方言列表
dialects="combined"

# 训练配置
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=2026
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp  # 或 deepspeed

# ==================== 训练流程 ====================

# Stage 0: 数据准备（添加 instruct 文件）
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "=========================================="
  echo "Stage 0: 数据准备 - 生成 instruct 文件"
  echo "=========================================="
  
  for x in ${dialects}; do
    echo "处理 ${x}..."
    data_path="${data_dir}/${x}"
    
    if [ ! -f "${data_path}/wav.scp" ]; then
      echo "错误: ${data_path}/wav.scp 不存在，请先运行 generate_dialect_dataset.py"
      exit 1
    fi
    
    # 生成 instruct 文件（CosyVoice3 需要）
    if [ ! -f "${data_path}/instruct" ]; then
      echo "生成 ${data_path}/instruct..."
      
      # 根据方言类型设置不同的指令
      if [ "${x}" == "hunan" ]; then
        instruct_text="Please speak in Hunanese dialect (Changsha accent).<|endofprompt|>"
      elif [ "${x}" == "henan" ]; then
        instruct_text="Please speak in Henanese dialect (Henan accent).<|endofprompt|>"
      else
        instruct_text="You are a helpful assistant.<|endofprompt|>"
      fi
      
      # 从 text 文件生成 instruct 文件
      awk -v inst="${instruct_text}" '{print $1, inst}' "${data_path}/text" > "${data_path}/instruct"
      echo "生成完成: $(wc -l < ${data_path}/instruct) 条"
    else
      echo "instruct 文件已存在，跳过"
    fi
  done
fi

# Stage 1: 提取 Speaker Embedding
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "=========================================="
  echo "Stage 1: 提取 Speaker Embedding"
  echo "=========================================="
  echo "生成 utt2embedding.pt 和 spk2embedding.pt"
  
  for x in ${dialects}; do
    echo "处理 ${x}..."
    
    if [ -f "${data_dir}/${x}/utt2embedding.pt" ]; then
      echo "已存在，跳过"
      continue
    fi
    
    python ../../tools/extract_embedding.py \
      --dir "${data_dir}/${x}" \
      --onnx_path "${pretrained_model_dir}/campplus.onnx" \
      --num_thread 4
  done
fi

# Stage 2: 提取 Speech Token
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "=========================================="
  echo "Stage 2: 提取 Speech Token"
  echo "=========================================="
  echo "生成 utt2speech_token.pt"
  echo "注意: 此步骤需要 GPU 支持"
  
  for x in ${dialects}; do
    echo "处理 ${x}..."
    
    if [ -f "${data_dir}/${x}/utt2speech_token.pt" ]; then
      echo "已存在，跳过"
      continue
    fi
    
    python ../../tools/extract_speech_token.py \
      --dir "${data_dir}/${x}" \
      --onnx_path "${pretrained_model_dir}/speech_tokenizer_v3.onnx" \
      --num_thread 4
  done
fi

# Stage 3: 生成 Parquet 格式数据
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "=========================================="
  echo "Stage 3: 生成 Parquet 格式数据"
  echo "=========================================="
  
  for x in ${dialects}; do
    echo "处理 ${x}..."
    
    parquet_dir="${data_dir}/${x}/parquet"
    if [ -d "${parquet_dir}" ] && [ -f "${parquet_dir}/data.list" ]; then
      echo "已存在，跳过"
      continue
    fi
    
    mkdir -p "${parquet_dir}"
    
    python ../../tools/make_parquet_list.py \
      --num_utts_per_parquet 500 \
      --num_processes 4 \
      --instruct \
      --src_dir "${data_dir}/${x}" \
      --des_dir "${parquet_dir}"
  done
fi

# Stage 4: 合并数据列表
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "=========================================="
  echo "Stage 4: 合并数据列表"
  echo "=========================================="
  
  # 创建训练数据列表
  train_list="${data_dir}/train.data.list"
  > "${train_list}"
  
  for x in ${dialects}; do
    cat "${data_dir}/${x}/parquet/data.list" >> "${train_list}"
  done
  
  echo "训练数据列表: ${train_list}"
  echo "共 $(wc -l < ${train_list}) 个 parquet 文件"
  
  # 如果有验证集，可以类似处理
  # cat ${data_dir}/{dev-hunan,dev-henan}/parquet/data.list > ${data_dir}/dev.data.list
fi

# Stage 5: 训练模型
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "=========================================="
  echo "Stage 5: 训练模型"
  echo "=========================================="
  
  train_list="${data_dir}/train.data.list"
  
  if [ ! -f "${train_list}" ]; then
    echo "错误: ${train_list} 不存在，请先运行 stage 4"
    exit 1
  fi
  
  echo "训练配置:"
  echo "  - GPU: ${CUDA_VISIBLE_DEVICES}"
  echo "  - 训练引擎: ${train_engine}"
  echo "  - 数据: ${train_list}"
  
  if [ $train_engine == 'deepspeed' ]; then
    echo "  - DeepSpeed配置: ./conf/ds_stage2.json"
    echo "注意: DeepSpeed 有自己的优化器配置，如需修改请编辑 conf/ds_stage2.json"
  fi
  
  # 训练 LLM, Flow, HiFiGAN
  # 注意：对于微调，通常只需要训练 LLM 即可
  for model in llm; do  # 可以添加 flow hifigan
    echo ""
    echo "训练 ${model}..."
    
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data "${train_list}" \
      --cv_data "${train_list}" \
      --qwen_pretrain_path "${pretrained_model_dir}/CosyVoice-BlankEN" \
      --model $model \
      --checkpoint "${pretrained_model_dir}/${model}.pt" \
      --model_dir "$(pwd)/exp/dialect/${model}/${train_engine}" \
      --tensorboard_dir "$(pwd)/tensorboard/dialect/${model}/${train_engine}" \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# Stage 6: 模型平均（可选）
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "=========================================="
  echo "Stage 6: 模型平均"
  echo "=========================================="
  
  for model in llm; do
    decode_checkpoint="$(pwd)/exp/dialect/${model}/${train_engine}/${model}.pt"
    echo "最终模型: $decode_checkpoint"
    
    python ../../cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path "$(pwd)/exp/dialect/${model}/${train_engine}" \
      --num ${average_num} \
      --val_best
  done
fi

# Stage 7: 导出模型（可选）
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "=========================================="
  echo "Stage 7: 导出模型"
  echo "=========================================="
  
  # 将训练好的模型复制到推理目录
  model_dir="$(pwd)/exp/dialect/inference"
  mkdir -p "${model_dir}"
  
  echo "复制基础模型文件..."
  cp "${pretrained_model_dir}/campplus.onnx" "${model_dir}/"
  cp "${pretrained_model_dir}/speech_tokenizer_v3.onnx" "${model_dir}/"
  cp "${pretrained_model_dir}/CosyVoice-BlankEN" "${model_dir}/" -r 2>/dev/null || true
  
  echo "复制训练好的模型..."
  cp "$(pwd)/exp/dialect/llm/${train_engine}/llm.pt" "${model_dir}/" 2>/dev/null || true
  cp "${pretrained_model_dir}/flow.pt" "${model_dir}/" 2>/dev/null || true
  cp "${pretrained_model_dir}/hifigan.pt" "${model_dir}/" 2>/dev/null || true
  
  echo "模型已导出到: ${model_dir}"
  
  # 可选：导出 JIT/ONNX 格式
  # python ../../cosyvoice/bin/export_jit.py --model_dir "${model_dir}"
  # python ../../cosyvoice/bin/export_onnx.py --model_dir "${model_dir}"
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 查看 TensorBoard: tensorboard --logdir tensorboard/dialect"
echo "2. 测试推理: 参考 CosyVoice/example.py"
echo ""
