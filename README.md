# Solving Temporal Constraint QA with LLMs

## Installation

```bash
# Clone the repository
conda create -n temporal python=3.10
cd temporal_llms
pip install -r requirements.txt
cd ./alignment-handbook/
python -m pip install .
cd ..
```

### Additional step if CUDA Toolkit is missing

```bash
conda install -c conda-forge cudatoolkit-dev -y
```

## Usage

1. Run inference on CotempQA dataset:

```bash
# example
python inference.py \
    --model_name meta-llama/Llama-3.1-8B \
    --data_path data/cotempqa/mix.json \
    --mode default \
    --output_dir results/Cotempqa/evaluation_outputs/ \
    --evaluate_result_dir results/Cotempqa/evaluation_results/
```

2. Train temporal relation classifier on CotempQA dataset:

```bash
# example
python cotempqa_roberta_classification.py \
    --device 0 \
    --num_labels 4 \
    --max_len 128 \
    --batch_size 32 \
    --num_epochs 10 \
    --model_name roberta-base \
    --lr 1e-4 \
    --storage_dir models/bert_models
```

3. Create SFT dataset for CotempQA with Input/Output format (ensure CSV is formatted correctly): 

```bash
# example
python create_sft_dataset.py \
    --csv_path data/cotempqa/dataset_with_labels.csv \
    --input_col question \
    --output_col answer \
    --output_dir data/cotempqa/sft_dataset
```

4. Create SFT dataset for CotempQA with Input/Output format (ensure CSV is formatted correctly):

```bash
# example
python create_sft_dataset.py \
    --csv_path data/cotempqa/dataset_with_labels.csv \
    --input_col question \
    --output_col answer \
    --output_dir data/cotempqa/sft_dataset \
```

5. Create SFT dataset for CotempQA with Input/Reasoning/Output format (ensure CSV is formatted correctly):

```bash
# example
python create_sft_dataset.py \
    --csv_path data/cotempqa/dataset_with_labels.csv \
    --input_col question \
    --output_col answer \
    --output_dir data/cotempqa/sft_dataset_with_reasoning \
    --include_reasoning \
    --reasoning_col label
```

### Push models to Huggingface and load that dataset in the SFT scripts accordingly.

5. SFT on Cotempqa example using default settings with QLoRA for the CotempQA SFT dataset (input/output)

```bash
python scripts/cotempqa_llama_sft_vanilla.py \
    --dataset_path data/cotempqa/sft_dataset \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --output_dir models/llama3-8b-sft-adapter \
    # --hf_token YOUR_HF_TOKEN  # Optional: if needed for model download or push
    # --wandb_token YOUR_WANDB_TOKEN # Optional: for logging
```

6. SFT on Cotempqa example using default settings with QLoRA for the CotempQA SFT dataset (input/output)

```bash
python scripts/cotempqa_llama_sft_reasoning.py \
    --dataset_path data/cotempqa/sft_dataset \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --output_dir models/llama3-8b-sft-adapter_reasoning \
    # --hf_token YOUR_HF_TOKEN  # Optional: if needed for model download or push
    # --wandb_token YOUR_WANDB_TOKEN # Optional: for logging
```

<!-- 6. SFT on Cotempqa example adjusting parameters

```bash
python run_sft.py \
    --dataset_path path/to/save/dataset_iro \
    --output_dir ./llama3_8b_chat_adapter_iro \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --max_seq_len 2048 \
    --lora_r 64 \
    --lora_alpha 128 \
    --disable_flash_attention # Uncomment if flash attention causes issues \
    --disable_qlora # Uncomment to train without 4-bit quantization (needs more VRAM)
``` -->
