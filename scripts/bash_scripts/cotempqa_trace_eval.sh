#!/bin/bash

#SBATCH --cpus-per-task=32     # number of TASKS
#SBATCH -N 1     # keep all tasks on the same node
#SBATCH --mem=80G     # request 120 GB of memory
#SBATCH --partition public
#SBATCH --gres=gpu:a100:1
#SBATCH --time 1:00:00 

# module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate temporal
conda install anaconda::nltk
conda install conda-forge::rouge-score

git checkout main

# cd /home/sbhambr1/research/temporal_llms/temporal_llms/

python scripts/cotempqa_trace_eval.py