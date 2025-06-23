#!/usr/bin/env python3
"""
Script to analyze the length of 'r1_trace' entries in JSON files after tokenization
using the Llama-3.1-8b model tokenizer.

This script calculates the minimum, maximum, and average token lengths for all
'r1_trace' entries in the specified JSON file.
"""

import json
import argparse
import statistics
from typing import List, Dict, Any
from transformers import AutoTokenizer
import sys
import os

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file. Handles both regular JSON and JSONL formats.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    data = []
    
    try:
        # First try to load as regular JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                # JSON array format
                data = json.loads(content)
            else:
                # Try JSONL format (one JSON object per line)
                f.seek(0)
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
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)
        
    return data

def get_tokenizer():
    """
    Load the Llama-3.1-8b tokenizer.
    
    Returns:
        The loaded tokenizer
    """
    try:
        # Use the official Llama-3.1-8b model identifier
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Note: You may need to request access to the Llama model or use an alternative tokenizer.")
        print("Trying alternative tokenizer...")
        
        # Fallback to a publicly available tokenizer with similar characteristics
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            print("Using DialoGPT tokenizer as fallback.")
            return tokenizer
        except Exception as e2:
            print(f"Error loading fallback tokenizer: {e2}")
            sys.exit(1)

def tokenize_and_count(text: str, tokenizer) -> int:
    """
    Tokenize text and return the number of tokens.
    
    Args:
        text: Text to tokenize
        tokenizer: The tokenizer to use
        
    Returns:
        Number of tokens
    """
    if not isinstance(text, str):
        # Convert to string if it's not already
        text = str(text)
    
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def analyze_trace_lengths(data: List[Dict[str, Any]], tokenizer) -> Dict[str, float]:
    """
    Analyze the token lengths of 'r1_trace' entries.
    
    Args:
        data: List of data entries
        tokenizer: The tokenizer to use
        
    Returns:
        Dictionary with min, max, and average lengths
    """
    trace_lengths = []
    
    for i, entry in enumerate(data):
        if 'r1_trace' not in entry:
            print(f"Warning: Entry {i} missing 'r1_trace' field")
            continue
            
        r1_trace = entry['r1_trace']
        if r1_trace is None:
            print(f"Warning: Entry {i} has null 'r1_trace' field")
            continue
            
        try:
            token_count = tokenize_and_count(r1_trace, tokenizer)
            trace_lengths.append(token_count)
            
            # Print progress for large datasets
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} entries...")
                
        except Exception as e:
            print(f"Warning: Error processing entry {i}: {e}")
            continue
    
    if not trace_lengths:
        print("Error: No valid 'r1_trace' entries found to analyze.")
        sys.exit(1)
    
    return {
        'min_length': min(trace_lengths),
        'max_length': max(trace_lengths),
        'avg_length': statistics.mean(trace_lengths),
        'median_length': statistics.median(trace_lengths),
        'total_entries': len(trace_lengths),
        'total_entries_in_file': len(data)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze token lengths of 'r1_trace' entries in JSON files"
    )
    parser.add_argument(
        'json_file',
        help='Path to the JSON file to analyze'
    )
    parser.add_argument(
        '--output',
        help='Output file to save results (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    print("Loading JSON data...")
    data = load_json_data(args.json_file)
    print(f"Loaded {len(data)} entries from {args.json_file}")
    
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    print("Tokenizer loaded successfully")
    
    print("Analyzing trace lengths...")
    results = analyze_trace_lengths(data, tokenizer)
    
    # Print results
    print("\n" + "="*50)
    print("TRACE LENGTH ANALYSIS RESULTS")
    print("="*50)
    print(f"File analyzed: {args.json_file}")
    print(f"Total entries in file: {results['total_entries_in_file']}")
    print(f"Valid r1_trace entries analyzed: {results['total_entries']}")
    print(f"Minimum token length: {results['min_length']}")
    print(f"Maximum token length: {results['max_length']}")
    print(f"Average token length: {results['avg_length']:.2f}")
    print(f"Median token length: {results['median_length']:.2f}")
    print("="*50)
    
    # Save results if output file specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
