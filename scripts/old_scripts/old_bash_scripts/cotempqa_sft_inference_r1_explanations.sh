#!/bin/bash

#SBATCH --cpus-per-task=32     # number of TASKS
#SBATCH -N 1     # keep all tasks on the same node
#SBATCH --mem=80G     # request 120 GB of memory
#SBATCH --partition general
#SBATCH --gres=gpu:a100:1
#SBATCH --time 4:00:00 

# module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate temporal

git checkout r1/cotemp

# cd /home/sbhambr1/research/temporal_llms/temporal_llms/

# model_id: meta-llama/Llama-3.2-1B-Instruct ; meta-llama/Llama-3.2-3B-Instruct ; meta-llama/Llama-3.1-8B-Instruct ; mistralai/Mistral-7B-Instruct-v0.3 ; google/gemma-3-1b-it ; Qwen/Qwen3-4B ; Qwen/Qwen3-1.7B ; Qwen/Qwen3-8B

modes=("default")
with_reasoning="False"

model_id=$1
model_name="${model_id#*/}"

for mode in "${modes[@]}"; do
    python scripts/cotempqa_sft_inference_r1.py \
        --model_name "${model_id}" \
        --adapter_path "cotempqa/${model_name}-sft-adapter-reasoning-r1-explanations" \
        --mode "$mode"
done