#!/bin/bash

device=1
num_labels=4
max_len=128
batch_size=128
num_epochs=10
model_name="roberta-large"
lr=1e-4
storage_dir="$(pwd)/models/bert_models"

# Add your code here to use the variables
python scripts/cotempqa_roberta_classification.py \
    --device $device \
    --num_labels $num_labels \
    --max_len $max_len \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --model_name $model_name \
    --lr $lr \
    --storage_dir $storage_dir \