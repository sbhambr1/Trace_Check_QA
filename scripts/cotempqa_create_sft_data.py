import argparse
import pandas as pd
from datasets import Dataset
import os
import sys
import warnings
from datasets import Dataset

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_sft_dataset(csv_path, input_col, output_col, output_dir, include_reasoning=False, reasoning_col=None):
    """
    Creates a Hugging Face dataset CSV from a CSV file for SFT.

    Args:
        csv_path (str): Path to the input CSV file.
        input_col (str): Name of the column containing the input/prompt.
        output_col (str): Name of the column containing the desired output/response.
        output_dir (str): Directory where the dataset will be saved.
        include_reasoning (bool): If True, include the reasoning trace. Defaults to False.
        reasoning_col (str, optional): Name of the column containing the reasoning trace.
                                      Required if include_reasoning is True. Defaults to None.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
            formatted_text = {
                "content": input_text,
                "role": "user"
            }, {
                "content": reasoning_text,
                "role": "assistant"
            }, {
                "content": output_text,
                "role": "assistant"
            }
            
        else:
            formatted_text = {
                "content": input_text,
                "role": "user"
            }, {
                "content": output_text,
                "role": "assistant"
            }

        data_list.append(list(formatted_text))
    # Create a dataset CSV file from data_list
    dataset_csv = df.copy()
    dataset_csv["messages"] = data_list
    dataset_csv_path = os.path.join(output_dir, "formatted_dataset.csv")
    dataset_csv.to_csv(dataset_csv_path, index=False)
    train_csv = dataset_csv.sample(frac=0.8, random_state=42)
    test_csv = dataset_csv.drop(train_csv.index)
    train_csv.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_csv.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Created dataset with {len(dataset_csv)} examples.")

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


