#!/bin/bash

#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH --mem=80G
#SBATCH --partition general
#SBATCH --gres=gpu:a100:1
#SBATCH --time 4:00:00

# module load cuda-11.7.0-gcc-11.2.0

eval "$(conda shell.bash hook)"
conda activate temporal

model_id=$1
dataset_type=$2

if [ -z "$model_id" ]; then
    echo "Usage: $0 <model_id> [max_new_tokens]"
    exit 1
fi

model_name="${model_id#*/}"
adapter_path="models/openthoughts/${model_name}-sft-adapter-${dataset_type}/final_adapter"

# Inference script for AIME2024 datasets (I and II)
python scripts/openthoughts_sft_inference_AIME2024.py \
  --model_name "$model_id" \
  --adapter_path "$adapter_path" \
  --max_new_tokens 32768
