#!/bin/bash

#SBATCH --cpus-per-task=32     # number of TASKS
#SBATCH -N 1     # keep all tasks on the same node
#SBATCH --mem=80G     # request 80 GB of memory
#SBATCH --partition general
#SBATCH --gres=gpu:a100:1
#SBATCH --time 4:00:00 

# module load cuda-11.7.0-gcc-11.2.0 

eval "$(conda shell.bash hook)"
conda activate temporal

git checkout r1/cotemp

# cd /home/sbhambr1/research/temporal_llms/temporal_llms/

wandb_token=$WANDB_API_KEY

# model_id: meta-llama/Llama-3.2-1B-Instruct (60 mins) ; meta-llama/Llama-3.2-3B-Instruct ; meta-llama/Llama-3.1-8B-Instruct ; mistralai/Mistral-7B-Instruct-v0.3 ; google/gemma-3-1b-it ; Qwen/Qwen3-4B ; Qwen/Qwen3-1.7B ; Qwen/Qwen3-8B

model_id=$1
dataset_type=$2

# Default to r1_trace if no dataset type provided
if [ -z "$dataset_type" ]; then
    dataset_type="r1_trace"
fi

# Validate dataset type
if [[ ! "$dataset_type" =~ ^(r1_trace|explanation|summary|no_reasoning|perturbed_reasoning)$ ]]; then
    echo "Error: Invalid dataset_type '$dataset_type'. Must be one of: r1_trace, explanation, summary, no_reasoning, perturbed_reasoning"
    exit 1
fi

model_name="${model_id#*/}"

echo "Training model: $model_id"
echo "Dataset type: $dataset_type"
echo "Model name: $model_name"

python scripts/openthoughts_sft_training.py \
    --model_id "$model_id" \
    --dataset_type "$dataset_type" \
    --expt_name "${model_name}-sft-openthoughts-${dataset_type}" \
    --output_dir "openthoughts/${model_name}-sft-adapter-${dataset_type}" \
    --wandb_token $wandb_token \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 1e-5 \
    --max_seq_len 1024 \
    --lora_r 32 \
    --lora_alpha 64
