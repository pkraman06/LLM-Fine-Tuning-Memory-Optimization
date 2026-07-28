#!/usr/bin/env bash
# Launch distributed QLoRA fine-tuning across all local GPUs using FSDP.
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-2-7b-hf"}
TRAIN_JSON=${TRAIN_JSON:-"./cuda.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/cuad-qlora-fsdp"}

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

accelerate launch --config_file configs/fsdp_config.yaml src/train.py \
  --model_name "${MODEL_NAME}" \
  --train_json "${TRAIN_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --bf16

