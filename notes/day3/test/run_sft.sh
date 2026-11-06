#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 \
  training/supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen3.5-0.8B \
  --train_file_dir ./notes/day3/test/data/train \
  --validation_file_dir ./notes/day3/test/data/dev \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 24 \
  --max_eval_samples 10 \
  --model_max_length 256 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_steps 1 \
  --logging_strategy steps \
  --logging_steps 1 \
  --eval_strategy steps \
  --eval_steps 2 \
  --save_strategy steps \
  --save_steps 3 \
  --save_total_limit 2 \
  --gradient_accumulation_steps 4 \
  --preprocessing_num_workers 1 \
  --output_dir ./notes/day3/test/artifacts/medaegis_sft_smoke \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --gradient_checkpointing True \
  --flash_attn False \
  --report_to tensorboard \
  --seed 42 \
  --cache_dir ./cache
