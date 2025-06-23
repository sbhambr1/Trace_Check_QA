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

# Usage:
# sbatch scripts/bash_scripts/openthoughts_sft_inference.sh \
#   <model_id> <dataset_type> [output_dir]

model_id=$1
dataset_type=$2
output_dir=${3:-results/OpenThoughts/inference_outputs/}

if [ -z "$model_id" ] || [ -z "$dataset_type" ]; then
    echo "Usage: $0 <model_id> <dataset_type> [output_dir]"
    exit 1
fi

model_name="${model_id#*/}"
adapter_path="models/openthoughts/${model_name}-sft-adapter-${dataset_type}/final_adapter"

python scripts/openthoughts_sft_inference.py \
    --model_name "$model_id" \
    --adapter_path "$adapter_path" \
    --output_dir "$output_dir" \
    --max_new_tokens 32768
