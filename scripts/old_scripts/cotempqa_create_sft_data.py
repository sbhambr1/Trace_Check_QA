import os
import sys
import ast
import warnings
import argparse
import pandas as pd


warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_sft_dataset(csv_path, input_col, output_col, output_dir, include_reasoning=False, reasoning_col=None, include_facts=False, facts_col=None):
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
            if include_facts:
                facts = ast.literal_eval(row[facts_col]) if pd.notna(row[facts_col]) else []
                answers = ast.literal_eval(row[output_col]) if pd.notna(row[output_col]) else []
                answer_facts = [fact for fact in facts if any(answer in fact for answer in answers)]
                facts_text = "I need to use the following facts to answer the question: " + str(answer_facts)
                reasoning_text = reasoning_text + "" + facts_text
            formatted_text = {
                "content": input_text,
                "role": "user"
            }, {
                "content": "<think>" + reasoning_text + "</think>" + " " + "<answer>" + output_text + "</answer>",
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
    
    print(f"Created dataset with {len(dataset_csv)} examples.")
    
    categories = ['overlap', 'during', 'mix', 'equal']
    for category in categories:
        category_train_csv = dataset_csv[dataset_csv["reasoning"].str.contains(category, case=False)]
        category_test_csv = category_train_csv.sample(frac=0.2, random_state=42)
        category_train_csv = category_train_csv.drop(category_test_csv.index)
        
        category_train_csv.to_csv(os.path.join(output_dir, f"{category}_train.csv"), index=False)
        category_test_csv.to_csv(os.path.join(output_dir, f"{category}_test.csv"), index=False)
        
def merge_category_csvs(output_dir):
    categories = ['overlap', 'during', 'mix', 'equal']
    train_csvs = []
    test_csvs = []

    for category in categories:
        train_csv_path = os.path.join(output_dir, f"{category}_train.csv")
        test_csv_path = os.path.join(output_dir, f"{category}_test.csv")

        train_csv = pd.read_csv(train_csv_path)
        test_csv = pd.read_csv(test_csv_path)

        train_csvs.append(train_csv)
        test_csvs.append(test_csv)

    merged_train_csv = pd.concat(train_csvs)
    merged_test_csv = pd.concat(test_csvs)

    merged_train_csv.to_csv(os.path.join(output_dir, "merged_train.csv"), index=False)
    merged_test_csv.to_csv(os.path.join(output_dir, "merged_test.csv"), index=False)

    print("Merged train and test CSV files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Hugging Face dataset for SFT from a CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--input_col", type=str, required=True, help="List of column names for input/prompt.")
    parser.add_argument("--output_col", type=str, required=True, help="Column name for the desired output.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the formatted dataset.")
    parser.add_argument("--include_reasoning", action='store_true', help="Include reasoning trace in the format.")
    parser.add_argument("--reasoning_col", type=str, help="Column name for the reasoning trace (required if --include_reasoning is used).")
    parser.add_argument("--include_facts", action='store_true', help="Include facts in the reasoning trace.")
    parser.add_argument("--facts_col", type=str, help="Column name for the facts (required if --include_facts is used).")

    args = parser.parse_args()

    create_sft_dataset(
        csv_path=args.csv_path,
        input_col=args.input_col,
        output_col=args.output_col,
        output_dir=args.output_dir,
        include_reasoning=args.include_reasoning,
        reasoning_col=args.reasoning_col,
        include_facts=args.include_facts,
        facts_col=args.facts_col
    )

    merge_category_csvs(args.output_dir)


    
