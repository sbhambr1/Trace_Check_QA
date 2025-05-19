# Invesitgating Trace-based Knowledge Distillation

## Installation

```bash
# Clone the repository
conda create -n trace_kd python=3.10
cd trace_kd
pip install -r requirements.txt
cd ..
```

### Additional step if CUDA Toolkit is missing

```bash
conda install -c conda-forge cudatoolkit-dev -y
```

## Usage

1. Run inference on bAbi QA dataset:

```bash
# example
python scripts/babiqa_inference.py \
    --model_name meta-llama/Llama-3.2-1B \
    --data_path data/babiqa/sft_dataset_chat_template/test.csv \
    --mode default \
    --output_dir results/babiqa/evaluation_outputs/default/ \
    --evaluate_result_dir results/babiqa/evaluation_results/default/
```

2. Create SFT dataset for bAbi QA with Input/Output format (ensure CSV is formatted correctly): 

```bash
# example
python babiqa_create_sft_dataset.py \
    --csv_path data/babiqa/dataset_with_labels.csv \
    --input_col question \
    --output_col answer \
    --output_dir data/babiqa/sft_dataset_chat_template
```

3. Create SFT dataset for bAbi QA with Input/Reasoning/Output format (ensure CSV is formatted correctly):

```bash
# example
python babiqa_create_sft_dataset.py \
    --csv_path data/babiqa/dataset_with_labels.csv \
    --input_col question \
    --output_col answer \
    --output_dir data/babiqa/sft_dataset_reasoning_facts_chat_template \
    --include_reasoning \
    --reasoning_col label
```

### Push data to your Huggingface and load that dataset in the SFT scripts accordingly. (Data will be made public later on.)

4. SFT on bAbi QA example using default settings with QLoRA for the bAbi QA SFT dataset (input/output)

```bash
sbatch ./scripts/bash_scripts/babiqa_sft_vanilla.sh "meta-llama/Llama-3.2-1B-Instruct"
```

5. SFT on bAbi QA example using default settings with QLoRA for the bAbi QA SFT dataset with Temporal Relation and Facts in reasoning trace (input/reasoning + output) (gold labels + facts)

```bash
sbatch ./scripts/bash_scripts/babiqa_sft_reasoning_facts.sh "meta-llama/Llama-3.2-1B-Instruct"
```

6. Inference

```bash
# for SFT models without any reasoning trace
sbatch ./scripts/bash_scripts/babiqa_sft_inference.sh "meta-llama/Llama-3.2-1B-Instruct" False False

# for SFT models with reasoning trace with only temporal relation
sbatch ./scripts/bash_scripts/babiqa_sft_inference.sh "meta-llama/Llama-3.2-1B-Instruct" True True
```