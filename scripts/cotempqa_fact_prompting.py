import os
import ast
import sys
import json
import argparse
from datasets import load_from_disk, load_dataset, load_metric
from scripts.conversation import Conversation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


dataset = load_dataset("sbhambr1/cotempqa_for_sft_reasoning_facts", data_files={"train": "train.csv", "test": "test.csv"})

train_data = dataset['train'].to_pandas()
test_data = dataset['test'].to_pandas()

train_data['facts'] = train_data['facts'].apply(ast.literal_eval)
train_data['answer'] = train_data['answer'].apply(ast.literal_eval)

train_data['answer_facts'] = train_data.apply(lambda row: [fact for fact in row['facts'] if any(answer in fact for answer in row['answer'])], axis=1)

test_data['facts'] = test_data['facts'].apply(ast.literal_eval)
test_data['answer'] = test_data['answer'].apply(ast.literal_eval)

test_data['answer_facts'] = test_data.apply(lambda row: [fact for fact in row['facts'] if any(answer in fact for answer in row['answer'])], axis=1)

TASK_DESC = "This is a temporal reasoning question answering task where you are given a question, a set of facts, and the temporal relation asked in the question. Your task is to extract the relevant facts from the given set of facts that are necessary to answer the question. The relevant facts should be in the same format as the input facts. Please provide only the relevant facts without any additional information or explanation."

def create_prompt(data):
    facts = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(data['facts'])])
    answer_facts = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(data['answer_facts'])])
    
    prompt = f"""
        {TASK_DESC}
        Question: {data['question']}
        Facts:
        {facts}
        Relevant Facts:
        {answer_facts}
    """
    return prompt

metric = load_metric("accuracy")

def prompting(data):
    prompts = []
    for i in range(len(data)):
        prompt = create_prompt(data.iloc[i])
        prompts.append(prompt)
    return prompts

def evaluate(llm_model, prompts, answers):
    predictions = []
    for prompt in prompts:
        prediction = llm_model.get_response(prompt)
        prediction = prediction.strip()
        predictions.append(prediction)
    metric.add_batch(predictions=predictions, references=answers)
    results = metric.compute()
    return results

def save_results(llm_model_name, results):
    directory = "results/cotempqa"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{llm_model_name}_fact_retrieval.json")
    with open(filepath, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate LLM on CoTempQA Fact Retrieval")
    parser.add_argument("--llm_model_name", type=str, required=True, help="LLM model name")
    args = parser.parse_args()    
    
    llm_model = Conversation(args.llm_model_name, temp=0)
    train_prompts = prompting(train_data)
    test_prompts = prompting(test_data)
    
    train_results = evaluate(llm_model, train_prompts, ast.literal_eval(train_data['answer'].tolist()))
    test_results = evaluate(llm_model, test_prompts, ast.literal_eval(test_data['answer'].tolist()))
    
    save_results(f"{args.llm_model_name}_train", train_results)
    save_results(f"{args.llm_model_name}_test", test_results)