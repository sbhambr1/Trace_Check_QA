import os
import sys
import ast
import random
import warnings
import argparse
import pandas as pd


warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def change_category(text):
    """
    Changes the category of the reasoning trace in the text.
    
    Args:
        text (str): The reasoning trace text to be modified.
    
    Returns:
        str: The modified reasoning trace text with the category changed.
    """
    categories = ["single-supporting-fact","two-supporting-facts","two-arg-relations","counting","lists-sets","conjunction","time-reasoning","basic-deduction","basic-induction","positional-reasoning", "size-reasoning"]
    for category in categories:
        if category in text:
            new_category = random.choice([cat for cat in categories if cat != category])
            text = text.replace(category, new_category)
            break
    return text

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
            reasoning_text = change_category(reasoning_text)
            if include_facts:
                answer_facts = str(row[facts_col]) if pd.notna(row[facts_col]) else []
                
                try:
                    answer_facts_str = ast.literal_eval(answer_facts)
                except (ValueError, SyntaxError):
                    answer_facts_str = str(answer_facts).lower()
                # remove all punctuations from answer_fact_str except spaces
                
                for i in range(len(answer_facts_str)):
                    answer_facts_str[i] = answer_facts_str[i].lower()
                
                all_facts = ast.literal_eval(row['passages'])
                
                for i in range(len(all_facts)):
                    all_facts[i] = all_facts[i].lower()
                
                other_facts = [fact for fact in all_facts if fact not in answer_facts_str]
                
                # other_facts = [fact for fact in all_facts if answer_fact_str not in fact.lower()]
                random_facts = []
                if len(other_facts) > len(answer_facts_str):
                    num_facts = len(answer_facts_str)
                else:
                    num_facts = len(other_facts)
                random_facts = random.sample(other_facts, num_facts)
                
                facts_text = "I need to use the following text to answer the question: " + str(random_facts)
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
    if 'train' in csv_path:
        dataset_csv_path = os.path.join(output_dir, "train.csv")
    elif 'test' in csv_path:
        dataset_csv_path = os.path.join(output_dir, "test.csv")
    dataset_csv.to_csv(dataset_csv_path, index=False)
    
    print(f"Created dataset with {len(dataset_csv)} examples.")


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


    
