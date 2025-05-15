#!/bin/bash

#SBATCH --cpus-per-task=32     # number of TASKS
#SBATCH -N 1     # keep all tasks on the same node
#SBATCH --mem=80G     # request 120 GB of memory
#SBATCH --partition general
#SBATCH --gres=gpu:a100:1
#SBATCH --time 1:00:00 

# module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate temporal

git checkout babiqa

# cd /home/sbhambr1/research/temporal_llms/temporal_llms/

# model_id: meta-llama/Llama-3.2-1B-Instruct ; meta-llama/Llama-3.2-3B-Instruct ; meta-llama/Llama-3.1-8B-Instruct ; mistralai/Mistral-7B-Instruct-v0.3 ; google/gemma-3-1b-it ; Qwen/Qwen3-4B ; Qwen/Qwen3-1.7B ; Qwen/Qwen3-8B

# data_types=("mix" "equal" "during" "overlap")
# data_types=(")
# modes=("default" "few_shot_cot" "few_shot")
modes=("default")
with_reasoning="False"

model_id=$1
with_reasoning=$2
with_reasoning_facts=$3
model_name="${model_id#*/}"

added_name=""
adapter_name=""

if [ "$with_reasoning" = "True" ]; then
    added_name="with_reasoning/"
    adapter_name="-reasoning"
fi

if [ "$with_reasoning_facts" = "True" ]; then
    added_name="with_reasoning_facts/"
    adapter_name="-reasoning-facts"
fi

for mode in "${modes[@]}"; do
    python scripts/babiqa_sft_inference_fixed.py \
        --model_name "${model_id}" \
        --adapter_path "babiqa/${model_name}-sft-adapter${adapter_name}" \
        --mode "$mode"
done