#!/bin/bash

SBATCH -c 1     # number of TASKS
SBATCH -N 1     # keep all tasks on the same node
SBATCH --mem=100G     # request 120 GB of memory
SBATCH -p general
SBATCH --gres=gpu:a100:1
SBATCH -t 0-01:30:00 

module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate temporal

cd /home/sbhambr1/research/temporal_llms/temporal_llms/

wandb_token=$WANDB_API_KEY

# model_id: meta-llama/Llama-3.2-1B-Instruct ; meta-llama/Llama-3.2-3B-Instruct ; meta-llama/Llama-3.1-8B-Instruct ; mistralai/Mistral-7B-Instruct-v0.3 ; google/gemma-3-1b-it ; Qwen/Qwen3-4B ; Qwen/Qwen3-1.7B ; Qwen/Qwen3-8B

python scripts/cotempqa_sft_reasoning.py \
    --expt_name llama3.1-8b-sft-cotempqa-with_reasoning \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --output_dir models/llama3-8b-sft-adapter_reasoning \
    --wandb_token $wandb_token
