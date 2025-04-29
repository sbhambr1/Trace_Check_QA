#!/bin/bash

# SBATCH -c 1     # number of TASKS
# SBATCH -N 1     # keep all tasks on the same node
# SBATCH --mem=100G     # request 120 GB of memory
# SBATCH -p general
# SBATCH --gres=gpu:a100:1
# SBATCH -t 0-01:30:00 

eval "$(conda shell.bash hook)"
conda activate temporal

cd /home/sbhambr1/research/temporal_llms/temporal_llms/

wandb_token=$WANDB_API_KEY

python scripts/cotempqa_llama_sft_vanilla.py \
    --expt_name llama3.1-8b-sft-cotempqa \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --output_dir models/llama3-8b-sft-adapter \
    --wandb_token $wandb_token