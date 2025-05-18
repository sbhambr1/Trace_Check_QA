#!/bin/bash

# SBATCH -c 1     # number of TASKS
# SBATCH -N 1     # keep all tasks on the same node
# SBATCH --mem=100G     # request 120 GB of memory
# SBATCH -p general
# SBATCH --gres=gpu:a100:1
# SBATCH -t 0-01:30:00 

# module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate temporal

# model_id: meta-llama/Llama-3.2-1B-Instruct ; meta-llama/Llama-3.2-3B-Instruct ; meta-llama/Llama-3.1-8B-Instruct ; mistralai/Mistral-7B-Instruct-v0.3 ; google/gemma-3-1b-it ; google/gemma-3-4b-it ; Qwen/Qwen3-1.7B ; Qwen/Qwen3-4B ; Qwen/Qwen3-8B ; deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ; deepseek-ai/DeepSeek-R1-Distill-Qwen-7B ; deepseek-ai/DeepSeek-R1-Distill-Llama-8B

model_name=$1
device=$2

# modes=("default" "few_shot_cot" "few_shot")
modes=("default")

for mode in "${modes[@]}"; do
    CUDA_VISIBLE_DEVICES="$device" python scripts/marco_inference.py \
        --model_name "$model_name" \
        --data_path "data/marcoqa/sft_dataset_chat_template/test.csv" \
        --mode "$mode" \
        --output_dir "results/marcoqa/evaluation_outputs/${mode}/" \
        --evaluate_result_dir "results/marcoqa/evaluation_results/${mode}/"
done
