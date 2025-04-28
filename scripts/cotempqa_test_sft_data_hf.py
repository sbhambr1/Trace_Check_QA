import argparse
import pandas as pd
from datasets import Dataset, load_dataset
import os
import sys
import warnings

warnings.filterwarnings("ignore")

hf_dataset = load_dataset("sbhambr1/cotempqa_for_sft", data_files={"train": "train.csv", "test": "test.csv"})

print(hf_dataset)

hf_dataset_reasoning = load_dataset("sbhambr1/cotempqa_for_sft_reasoning", data_files={"train": "train.csv", "test": "test.csv"})

print(hf_dataset_reasoning)