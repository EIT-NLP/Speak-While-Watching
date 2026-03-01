#!/bin/bash

# Distributed training configuration
NPROC_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# Script paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# scripts -> qwen-vl-finetune -> code
PY_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="$PY_ROOT:$PYTHONPATH"

# Variant selection (origin / batch / group / gap / overlap / interleave)
export QWEN2_5_VL_VARIANT=${QWEN2_5_VL_VARIANT:-origin}

export VIDEO_MIN_PIXELS=78400
export FPS_MAX_FRAMES=60
export VIDEO_MAX_PIXELS=2408448

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=${QWEN2_5_VL_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}

# Training hyperparameters
lr=2e-5
batch_size=1
grad_accum_steps=16

# Training entry point
entry_file=qwenvl/train/train.py

# Dataset configuration
datasets=${QWEN2_5_VL_TRAIN_DATA:-$PY_ROOT/../dataset/train_3_5.jsonl}

# Output configuration
run_name="qwen2_5vl-pe-${QWEN2_5_VL_VARIANT}"
output_dir=./output/qwen2_5vl-pe-${QWEN2_5_VL_VARIANT}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to none"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
