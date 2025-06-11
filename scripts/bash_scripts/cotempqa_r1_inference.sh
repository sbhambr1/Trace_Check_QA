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

# data_types=("mix" "equal" "during" "overlap")
data_types=("mix" "overlap" "during")
modes=("default")

for data_type in "${data_types[@]}"; do
    for mode in "${modes[@]}"; do
        python scripts/cotempqa_r1_inference.py \
            --data_path "data/cotempqa/${data_type}.json" \
            --mode "$mode" \
            --output_dir "results/Cotempqa/evaluation_outputs/${data_type}_${mode}/" \
            --evaluate_result_dir "results/Cotempqa/evaluation_results/${data_type}_${mode}/"
    done
done
