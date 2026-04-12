# Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation

<h3 align="center">🎉 Accepted at <a href="https://2026.aclweb.org/">ACL 2026</a></h3>

<p align="center">
  <a href="https://github.com/sbhambr1/Trace_Check_QA"><img src="https://img.shields.io/badge/GitHub-Trace--Check--QA-black?logo=github"/></a>
  <a href="https://hf.co/collections/sbhambr1/trace-check-qa-datasets"><img src="https://img.shields.io/badge/🤗%20Datasets-HuggingFace-yellow"/></a>
</p>

---

## Installation

```bash
# Clone the repository
conda create -n trace_kd python=3.10
<<<<<<< HEAD
cd trace_kd
=======
conda activate trace_kd
cd Trace_Check_QA
>>>>>>> 525620a5061f1b7a2c8f7a55e0c8d6a4ea51fff6
pip install -r requirements.txt
cd ..
```

### Additional step if CUDA Toolkit is missing

```bash
conda install -c conda-forge cudatoolkit-dev -y
```

---

## Usage

1. Run inference on CotempQA dataset:

```bash
# example
python inference.py \
    --model_name meta-llama/Llama-3.2-1B \
    --data_path data/cotempqa/mix.json \
    --mode default \
    --output_dir results/Cotempqa/evaluation_outputs/ \
    --evaluate_result_dir results/Cotempqa/evaluation_results/
```

2. Create SFT dataset for CotempQA with Input/Output format (ensure CSV is formatted correctly): 

```bash
# example
python create_sft_dataset.py \
    --csv_path data/cotempqa/dataset_with_labels.csv \
    --input_col question \
    --output_col answer \
    --output_dir data/cotempqa/sft_dataset
```

3. Create SFT dataset for CotempQA with Input/Reasoning/Output format (ensure CSV is formatted correctly):

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

### Push data to your Huggingface and load that dataset in the SFT scripts accordingly (or use the links provided).

4. SFT on Cotempqa example using default settings with QLoRA for the CotempQA SFT dataset (input/output)

```bash
sbatch ./scripts/bash_scripts/cotempqa_sft_vanilla.sh "meta-llama/Llama-3.2-1B-Instruct"
```

5. SFT on Cotempqa example using default settings with QLoRA for the CotempQA SFT dataset with Temporal Relation and Facts in reasoning trace (input/reasoning + output) (gold labels + facts)

```bash
sbatch ./scripts/bash_scripts/cotempqa_sft_reasoning_facts.sh "meta-llama/Llama-3.2-1B-Instruct"
```

6. Inference

```bash
# for SFT models without any reasoning trace
sbatch ./scripts/bash_scripts/cotempqa_sft_inference.sh "meta-llama/Llama-3.2-1B-Instruct" False False

# for SFT models with reasoning trace with only temporal relation
sbatch ./scripts/bash_scripts/cotempqa_sft_inference.sh "meta-llama/Llama-3.2-1B-Instruct" True True
<<<<<<< HEAD
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{bhambri2025tracecheckqa,
  title     = {Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect
in Trace-Based Knowledge Distillation},
  author    = {Bhambri, Siddhant and Biswas, Upasana and Kambhampati, Subbarao},
  booktitle = {Proceedings of The 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026},
  url       = {https://github.com/sbhambr1/Trace_Check_QA}
}
```
