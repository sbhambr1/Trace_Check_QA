#!/usr/bin/env python
"""
Script to push datasets from data/OpenThoughts to Hugging Face Hub.
Usage: python scripts/push_openthoughts_to_hf.py --hf_token <your_token> --org <your_org>
"""
import os
import argparse
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder

DATA_DIR = "data/OpenThoughts"


def push_dataset_to_hf(dataset_path, hf_token, org, collection=None):
    dataset_name = os.path.basename(dataset_path)
    repo_id = f"{org}/{dataset_name}"
    print(f"Processing {dataset_path} -> {repo_id}")

    api = HfApi(token=hf_token)
    if api.repo_exists(repo_id, repo_type="dataset"):
        print(f"{repo_id} already exists on Hugging Face Hub. Skipping.")
        return

    # Find all csv files in the dataset_path
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {dataset_path}, skipping.")
        return

    # Load each csv file as a split
    splits = {}
    for csv_file in csv_files:
        split_name = os.path.splitext(csv_file)[0]
        split_path = os.path.join(dataset_path, csv_file)
        try:
            ds = Dataset.from_csv(split_path)
            if len(ds) == 0:
                print(f"Split '{split_name}' in {csv_file} is empty. Skipping this split.")
                continue
            splits[split_name] = ds
        except Exception as e:
            print(f"Error loading split '{split_name}' from {csv_file}: {e}. Skipping this split.")

    if not splits:
        print(f"No valid splits found in {dataset_path}, skipping dataset upload.")
        return

    dataset_dict = DatasetDict(splits)

    # Push to Hugging Face Hub
    dataset_dict.push_to_hub(repo_id, token=hf_token)
    print(f"Pushed {repo_id} to Hugging Face Hub.")

    # Add to collection if specified
    if collection:
        try:
            api.add_to_collection(
                collection_id=collection,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"Added {repo_id} to collection {collection}.")
        except Exception as e:
            print(f"Failed to add {repo_id} to collection {collection}: {e}")


def get_hf_token(cli_token=None):
    if cli_token:
        return cli_token
    # Try to get from environment
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # Try to read from ~/.bashrc
    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r") as f:
            for line in f:
                if "HF_TOKEN" in line and "export" in line:
                    # Example: export HF_TOKEN=xxxx
                    parts = line.strip().split("=")
                    if len(parts) == 2:
                        return parts[1].replace('"', '').replace("'", '')
    # Prompt user if not found
    print("Hugging Face token not found in CLI, environment, or ~/.bashrc. Please provide it with --hf_token.")
    exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', required=False, help='Hugging Face token')
    parser.add_argument('--org', required=True, help='Hugging Face organization or username')
    parser.add_argument('--collection', required=False, help='Hugging Face collection ID (org/collection_name)')
    args = parser.parse_args()

    hf_token = get_hf_token(args.hf_token)
    HfFolder.save_token(hf_token)

    for folder in os.listdir(DATA_DIR):
        dataset_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(dataset_path):
            push_dataset_to_hf(dataset_path, hf_token, args.org, args.collection)

if __name__ == "__main__":
    main()
