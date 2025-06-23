#!/usr/bin/env python3
"""
Script to create SFT datasets from OpenThoughts JSON files.

This script processes OpenThoughts JSON data and creates three different SFT datasets:
1. r1_trace dataset - using r1_trace as reasoning_text
2. explanation dataset - using explanation from explanation JSON as reasoning_text  
3. summary dataset - using r1_trace_summary from summary JSON as reasoning_text

The datasets are saved in CSV format in the data/OpenThoughts directory.
"""

import os
import sys
import json
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any

warnings.filterwarnings("ignore")

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL data from file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []
        
    return data

def create_sft_dataset(data: List[Dict[str, Any]], reasoning_key: str, dataset_name: str) -> pd.DataFrame:
    """
    Create SFT dataset from JSON data.
    
    Args:
        data: List of data entries
        reasoning_key: Key to use for reasoning text ('r1_trace', 'explanation', 'r1_trace_summary')
        dataset_name: Name of the dataset for logging
        
    Returns:
        DataFrame with SFT format
    """
    sft_data = []
    
    print(f"Creating {dataset_name} dataset using '{reasoning_key}' as reasoning text...")
    
    for i, entry in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        # Check required fields
        if 'input' not in entry:
            print(f"Warning: Entry {i} missing 'input' field")
            continue
            
        if 'prediction' not in entry:
            print(f"Warning: Entry {i} missing 'prediction' field")
            continue
            
        if reasoning_key not in entry:
            print(f"Warning: Entry {i} missing '{reasoning_key}' field")
            continue
            
        input_text = entry['input']
        reasoning_text = entry[reasoning_key]
        output_text = entry['prediction']
        
        # Skip if any field is None or empty
        if not input_text or not reasoning_text or not output_text:
            print(f"Warning: Entry {i} has empty required fields")
            continue
            
        # Format the conversation for SFT
        formatted_messages = [
            {
                "content": input_text,
                "role": "user"
            },
            {
                "content": f"<think>{reasoning_text}</think> <answer>{output_text}</answer>",
                "role": "assistant"
            }
        ]
        
        # Create row for DataFrame
        row = {
            'index': i,
            'question': input_text,
            'reasoning': reasoning_text,
            'answer': output_text,
            'messages': formatted_messages,
            'domain': entry.get('domain', 'unknown'),
            'source': entry.get('source', 'unknown')
        }
        
        sft_data.append(row)
    
    df = pd.DataFrame(sft_data)
    print(f"Created {dataset_name} dataset with {len(df)} entries")
    
    return df

def create_no_reasoning_sft_dataset(data: List[Dict[str, Any]], dataset_name: str) -> pd.DataFrame:
    """
    Create SFT dataset with no reasoning text (only problem and answer).
    """
    sft_data = []
    print(f"Creating {dataset_name} dataset with no reasoning text...")
    for i, entry in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        if 'input' not in entry or 'prediction' not in entry:
            continue
        input_text = entry['input']
        output_text = entry['prediction']
        if not input_text or not output_text:
            continue
        formatted_messages = [
            {"content": input_text, "role": "user"},
            {"content": f"<answer>{output_text}</answer>", "role": "assistant"}
        ]
        row = {
            'index': i,
            'question': input_text,
            'reasoning': '',
            'answer': output_text,
            'messages': formatted_messages,
            'domain': entry.get('domain', 'unknown'),
            'source': entry.get('source', 'unknown')
        }
        sft_data.append(row)
    df = pd.DataFrame(sft_data)
    print(f"Created {dataset_name} dataset with {len(df)} entries")
    return df

def create_perturbed_reasoning_sft_dataset(data: List[Dict[str, Any]], reasoning_key: str, dataset_name: str) -> pd.DataFrame:
    """
    Create SFT dataset with perturbed reasoning: each problem is paired with a reasoning_text from another problem (no original pairings, unique assignment).
    """
    import random
    sft_data = []
    print(f"Creating {dataset_name} dataset with perturbed reasoning text...")
    n = len(data)
    indices = list(range(n))
    # Generate a derangement (no element in original position)
    def derangement(n):
        while True:
            perm = indices[:]
            random.shuffle(perm)
            if all(i != perm[i] for i in range(n)):
                return perm
    perm = derangement(n)
    for i, entry in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        if 'input' not in entry or 'prediction' not in entry or reasoning_key not in entry:
            continue
        input_text = entry['input']
        output_text = entry['prediction']
        if not input_text or not output_text:
            continue
        # Get perturbed reasoning from another entry
        perturbed_reasoning = data[perm[i]][reasoning_key]
        formatted_messages = [
            {"content": input_text, "role": "user"},
            {"content": f"<think>{perturbed_reasoning}</think> <answer>{output_text}</answer>", "role": "assistant"}
        ]
        row = {
            'index': i,
            'question': input_text,
            'reasoning': perturbed_reasoning,
            'answer': output_text,
            'messages': formatted_messages,
            'domain': entry.get('domain', 'unknown'),
            'source': entry.get('source', 'unknown')
        }
        sft_data.append(row)
    df = pd.DataFrame(sft_data)
    print(f"Created {dataset_name} dataset with {len(df)} entries")
    return df

def split_and_save_dataset(df: pd.DataFrame, output_dir: str, dataset_name: str, test_ratio: float = 0.2):
    """
    Split dataset into train/test and save as CSV files.
    
    Args:
        df: DataFrame to split and save
        output_dir: Output directory
        dataset_name: Name of the dataset
        test_ratio: Ratio of test data (default 0.2)
    """
    if df.empty:
        print(f"Warning: {dataset_name} dataset is empty, skipping save")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into train and test
    test_df = df.sample(frac=test_ratio, random_state=42)
    train_df = df.drop(test_df.index)
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved {dataset_name} dataset:")
    print(f"  Train: {len(train_df)} entries -> {train_path}")
    print(f"  Test: {len(test_df)} entries -> {test_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create SFT datasets from OpenThoughts JSON files"
    )
    parser.add_argument(
        '--base-json',
        default='results/OpenThoughts/deepseek_r1_with_explanation_gpt-4o-mini.json',
        help='Path to the base JSON file (contains r1_trace and explanation)'
    )
    parser.add_argument(
        '--summary-json', 
        default='results/OpenThoughts/deepseek_r1_with_summary_gpt-4o-mini.json',
        help='Path to the summary JSON file (contains r1_trace_summary)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/OpenThoughts',
        help='Base output directory for datasets'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0,
        help='Ratio of test data (default: 0)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['r1_trace', 'explanation', 'summary', 'no_reasoning', 'perturbed_reasoning'],
        default=['r1_trace', 'explanation', 'summary', 'no_reasoning', 'perturbed_reasoning'],
        help='Which datasets to create (default: all)'
    )
    
    args = parser.parse_args()
    
    print("OpenThoughts SFT Dataset Creator")
    print("=" * 50)
    
    # Dataset configurations
    dataset_configs = {
        'r1_trace': {
            'file': args.base_json,
            'reasoning_key': 'r1_trace',
            'output_subdir': 'sft_dataset_r1_traces',
            'description': 'R1 Trace Dataset',
            'type': 'standard'
        },
        'explanation': {
            'file': args.base_json,
            'reasoning_key': 'explanation', 
            'output_subdir': 'sft_dataset_explanations',
            'description': 'Explanation Dataset',
            'type': 'standard'
        },
        'summary': {
            'file': args.summary_json,
            'reasoning_key': 'r1_trace_summary',
            'output_subdir': 'sft_dataset_summaries', 
            'description': 'Summary Dataset',
            'type': 'standard'
        },
        'no_reasoning': {
            'file': args.base_json,
            'output_subdir': 'sft_dataset_no_reasoning',
            'description': 'No Reasoning Dataset',
            'type': 'no_reasoning'
        },
        'perturbed_reasoning': {
            'file': args.base_json,
            'reasoning_key': 'r1_trace',
            'output_subdir': 'sft_dataset_perturbed_reasoning',
            'description': 'Perturbed Reasoning Dataset',
            'type': 'perturbed'
        }
    }
    
    # Process each requested dataset
    for dataset_type in args.datasets:
        if dataset_type not in dataset_configs:
            print(f"Warning: Unknown dataset type '{dataset_type}', skipping")
            continue
            
        config = dataset_configs[dataset_type]
        
        print(f"\n--- Processing {config['description']} ---")
        
        # Load data
        print(f"Loading data from: {config['file']}")
        data = load_jsonl_data(config['file'])
        
        if not data:
            print(f"No data loaded from {config['file']}, skipping {dataset_type}")
            continue
            
        print(f"Loaded {len(data)} entries")
        
        # Create dataset
        if config['type'] == 'no_reasoning':
            df = create_no_reasoning_sft_dataset(data, config['description'])
        elif config['type'] == 'perturbed':
            df = create_perturbed_reasoning_sft_dataset(data, config['reasoning_key'], config['description'])
        else:
            df = create_sft_dataset(data, config['reasoning_key'], config['description'])
        
        if df.empty:
            print(f"No valid entries for {dataset_type}, skipping")
            continue
            
        # Save dataset
        output_dir = os.path.join(args.output_dir, config['output_subdir'])
        split_and_save_dataset(df, output_dir, config['description'], args.test_ratio)
    
    print("\n" + "=" * 50)
    print("Dataset creation completed!")
    
    # Print summary
    print("\nDataset Summary:")
    for dataset_type in args.datasets:
        if dataset_type in dataset_configs:
            config = dataset_configs[dataset_type]
            output_dir = os.path.join(args.output_dir, config['output_subdir'])
            train_path = os.path.join(output_dir, 'train.csv')
            test_path = os.path.join(output_dir, 'test.csv')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_size = len(pd.read_csv(train_path))
                test_size = len(pd.read_csv(test_path))
                print(f"  {config['description']}: {train_size} train, {test_size} test -> {output_dir}")

if __name__ == "__main__":
    main()
