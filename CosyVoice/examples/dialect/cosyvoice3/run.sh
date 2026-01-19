#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# Modified for Chinese dialect fine-tuning
. ./path.sh || exit 1;

stage=0
stop_stage=5

# 数据目录 (相对于 examples/dialect/cosyvoice3 目录)
data_dir=../../../../dataset/combined
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: 检查数据准备"
  echo "请确保已运行 prepare_training_data.py 生成 instruct 文件"
  if [ ! -f "${data_dir}/instruct" ]; then
    echo "错误: ${data_dir}/instruct 不存在"
    exit 1
  fi
  echo "数据检查通过: $(wc -l < ${data_dir}/wav.scp) 条音频"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Extract campplus speaker embedding"
  if [ -f "${data_dir}/utt2embedding.pt" ]; then
    echo "已存在，跳过"
  else
    tools/extract_embedding.py --dir ${data_dir} \
      --onnx_path ${pretrained_model_dir}/campplus.onnx
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Extract discrete speech token"
  if [ -f "${data_dir}/utt2speech_token.pt" ]; then
    echo "已存在，跳过"
  else
    tools/extract_speech_token.py --dir ${data_dir} \
      --onnx_path ${pretrained_model_dir}/speech_tokenizer_v3.onnx
  fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Prepare parquet format data"
  mkdir -p ${data_dir}/parquet
  if [ -f "${data_dir}/parquet/data.list" ]; then
    echo "已存在，跳过"
  else
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --instruct \
      --src_dir ${data_dir} \
      --des_dir ${data_dir}/parquet
  fi
fi

# train llm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=2026
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Run train. We only support llm training for now"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  
  # 使用 combined 目录的 parquet
  train_list=${data_dir}/parquet/data.list
  
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice3.yaml \
      --train_data ${train_list} \
      --cv_data ${train_list} \
      --qwen_pretrain_path ${pretrained_model_dir}/CosyVoice-BlankEN \
      --model $model \
      --checkpoint ${pretrained_model_dir}/${model}.pt \
      --model_dir `pwd`/exp/dialect/${model}/${train_engine} \
      --tensorboard_dir `pwd`/tensorboard/dialect/${model}/${train_engine} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/dialect/${model}/${train_engine}/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/dialect/${model}/${train_engine} \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
