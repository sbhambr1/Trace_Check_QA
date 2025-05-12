#!/bin/bash

#SBATCH --cpus-per-task=32     # number of TASKS
#SBATCH -N 1     # keep all tasks on the same node
#SBATCH --mem=80G     # request 120 GB of memory
#SBATCH --partition general
#SBATCH --gres=gpu:a100:1
#SBATCH --time 2:00:00 

# module load cuda-11.7.0-gcc-11.2.0 

eval "$(conda shell.bash hook)"
conda activate temporal

# cd /home/sbhambr1/research/temporal_llms/temporal_llms/

wandb_token=$WANDB_API_KEY

# model_id: meta-llama/Llama-3.2-1B-Instruct (60 mins) ; meta-llama/Llama-3.2-3B-Instruct ; meta-llama/Llama-3.1-8B-Instruct ; mistralai/Mistral-7B-Instruct-v0.3 ; google/gemma-3-1b-it ; Qwen/Qwen3-4B ; Qwen/Qwen3-1.7B ; Qwen/Qwen3-8B

model_id="meta-llama/Llama-3.2-1B-Instruct"
model_name="${model_id#*/}"

python scripts/cotempqa_sft_reasoning_facts.py \
    --model_id "$model_id" \
    --expt_name "${model_name}-sft-cotempqa-reasoning-facts-and-inference" \
    --output_dir "models/${model_name}-sft-adapter-reasoning-facts" \
    --wandb_token $wandb_token \
    --epochs 3
