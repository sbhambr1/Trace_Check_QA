# Solving Temporal Constraint QA with LLMs

## Installation

```bash
# Clone the repository
conda create -n temporal python=3.10
pip install -r requirements.txt
```

## Usage

1. Run inference on CotempQA dataset:

```bash
# example
python inference.py --model_name meta-llama/Llama-3.1-8B --data_path data/cotempqa/mix.json  --mode default --output_dir results/Cotempqa/evaluation_outputs/ --evaluate_result_dir results/Cotempqa/evaluation_results/
```

2. Train temporal relation classifier on CotempQA dataset:

```bash
# example
python cotempqa_roberta_classification.py --device 0 --num_labels 4 --max_len 128 --batch_size 32 --num_epochs 10 --model_name roberta-base --lr 1e-4 --storage_dir models/bert_models
```

