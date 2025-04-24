import argparse
import pandas as pd
from datasets import Dataset
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Llama 3.1 chat template structure adapted from [3]
# Template for Input/Output format
TEMPLATE_IO = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>"""

# Template for Input/Reasoning/Output format
TEMPLATE_IRO = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
**Reasoning:**
{reasoning}

**Final Answer:**
{output}<|eot_id|>"""

def create_sft_dataset(csv_path, input_col, output_col, output_dir, include_reasoning=False, reasoning_col=None):
    """
    Creates a Hugging Face dataset from a CSV file for SFT.

    Args:
        csv_path (str): Path to the input CSV file.
        input_col (str): Name of the column containing the input/prompt.
        output_col (str): Name of the column containing the desired output/response.
        output_dir (str): Directory where the dataset will be saved.
        include_reasoning (bool): If True, include the reasoning trace. Defaults to False.
        reasoning_col (str, optional): Name of the column containing the reasoning trace.
                                      Required if include_reasoning is True. Defaults to None.
    """
    # Validate arguments if reasoning is included
    if include_reasoning and not reasoning_col:
        raise ValueError("`reasoning_col` must be specified when `include_reasoning` is True.")

    # Load data from CSV
    try:
        csv_path = os.path.join(os.getcwd() + '/', csv_path)
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV from {csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Check if required columns exist
    required_cols = [input_col, output_col]
    if include_reasoning:
        required_cols.append(reasoning_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}")
        return

    data_list = []
    # Format data according to the chosen template
    print(f"Formatting data... Using reasoning trace: {include_reasoning}")
    for index, row in df.iterrows():
        input_text = str(row[input_col]) if pd.notna(row[input_col]) else ""
        output_text = str(row[output_col]) if pd.notna(row[output_col]) else ""

        if include_reasoning:
            reasoning_text = str(row[reasoning_col]) if pd.notna(row[reasoning_col]) else ""
            formatted_text = TEMPLATE_IRO.format(input=input_text, reasoning=reasoning_text, output=output_text)
        else:
            formatted_text = TEMPLATE_IO.format(input=input_text, output=output_text)

        # SFTTrainer expects a 'text' field containing the full formatted sequence [2][3]
        data_list.append({"text": formatted_text})

    # Create Hugging Face Dataset object [3]
    hf_dataset = Dataset.from_list(data_list)
    print(f"Created dataset with {len(hf_dataset)} examples.")

    # Save the dataset
    try:
        os.makedirs(output_dir, exist_ok=True)
        hf_dataset.save_to_disk(output_dir)
        print(f"Dataset successfully saved to {output_dir}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Hugging Face dataset for SFT from a CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--input_col", type=str, required=True, help="List of column names for input/prompt.")
    parser.add_argument("--output_col", type=str, required=True, help="Column name for the desired output.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the formatted dataset.")
    parser.add_argument("--include_reasoning", action='store_true', help="Include reasoning trace in the format.")
    parser.add_argument("--reasoning_col", type=str, help="Column name for the reasoning trace (required if --include_reasoning is used).")

    args = parser.parse_args()

    create_sft_dataset(
        csv_path=args.csv_path,
        input_col=args.input_col,
        output_col=args.output_col,
        output_dir=args.output_dir,
        include_reasoning=args.include_reasoning,
        reasoning_col=args.reasoning_col
    )

