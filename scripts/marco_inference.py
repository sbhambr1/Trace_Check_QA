import json
import pandas as pd
import argparse
from marco_config import *
import os
import sys
from vllm import LLM, SamplingParams
import warnings
from rich import print as rich_print
import builtins

builtins.print = rich_print

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def evaluate_marcoqa(model_name, data_path, mode, output_dir, evaluate_result_dir):
    """
    Evaluate the co-temporal reasoning capabilities of a model on a dataset.
    
    Parameters:
    model_name (str): Name of the model to evaluate.
    data_path (str): Path to the input dataset.
    mode (str): Evaluation mode (e.g., 'default', 'few_shot', 'few_shot_cot', 'few_shot_math_cot', 'default_with_trace').
    output_dir (str): Directory to save the evaluation outputs.
    evaluate_result_dir (str): Directory to save the evaluation results.
    """
    all_data = []
    data_csv_path = os.path.join(os.getcwd() + '/', data_path)
    df = pd.read_csv(data_csv_path, sep=',', encoding='utf-8')
    for _, row in df.iterrows():
        data = {
            'question': row['question'],
            'answers': row['answers'],
            'query': row['query'],
            'query_type': row['query_type'],
            'passages': row['passages'],
            'reasoning': row['reasoning'],
            'answer_passage': row['answer_passage'],
        }
        all_data.append(data)
    
    # with open(data_csv_path, 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         data = json.loads(line.strip())
    #         all_data.append(data)

    if mode == 'default':
        all_prompts = get_prompts(all_data, default_template)
        
    if model_name == 'gpt': # currently "gpt-3.5-turbo-1106"
        raise NotImplementedError("ChatGPT API is not supported yet.")
        # filename = os.path.basename(data_path)
        # output_dir = os.path.join(os.getcwd() + '/', output_dir)
        # output_path = os.path.join(output_dir, f"{mode}_{filename}")
        # with open(output_path, 'w', encoding='utf-8') as out_f:
        #     for cnt, prompt in enumerate(all_prompts):
        #         api_list = [os.environ["OPENAI_API_KEY"]]
        #         chatgpt(out_f, prompt, all_data[cnt], cnt, api_list) 
        
        # output_data = []
        # with open(output_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         output_data.append(json.loads(line))
        # result = evaluate_gpt(output_data)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=1, dtype="float16")
        sampling_params = SamplingParams(temperature=0, max_tokens=500) # use 500 for reasoning models
        all_outputs = llm.generate(all_prompts, sampling_params)
        all_outputs = [output.outputs[0].text for output in all_outputs]
        
        output_data = []
        for prompt, input_data, output in zip(all_prompts, all_data, all_outputs):
            prompt = 'Answer the question based on the context:' + prompt.split('Answer the question based on the context:')[-1]
            output_data.append({
                'input': prompt,
                'prediction': output,
                'gold': input_data['answers'],
                'question': input_data['question'],
                'query_type': input_data['query_type'],
                'passage': input_data['passages'],
                'reasoning': input_data['reasoning'],
                'answer_passage': input_data['answer_passage'],
            })
        
        filename = os.path.basename(data_path)
        output_dir = os.path.join(os.getcwd() + '/', output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sanitized_model_name = model_name.replace("/", "_")
        output_path = os.path.join(output_dir, f"{sanitized_model_name}_{mode}_{filename}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in output_data:
                json_data = json.dumps(data)
                f.write(json_data + '\n')

        result = evaluate_model(output_data, mode)
        
    evaluate_result_path = os.path.join(evaluate_result_dir, f"{sanitized_model_name}_{mode}_{filename}")
    evaluate_result_dir = os.path.join(os.getcwd() + '/', evaluate_result_dir)
    if not os.path.exists(evaluate_result_dir):
        os.makedirs(evaluate_result_dir)
        
    with open(evaluate_result_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(result)
        f.write(json_data + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Co-temporal datasets")
    parser.add_argument("--model_name", type=str, help="Path to the model")
    parser.add_argument("--data_path", type=str, help="Path to the dataset file")
    parser.add_argument("--mode", type=str, help="Method to evaluate the co-temporal ability of LLMs")
    parser.add_argument("--output_dir", type=str, help="Path to save the outputs")
    parser.add_argument("--evaluate_result_dir", type=str, help="Path to save the evaluation result")
    
    args = parser.parse_args()

    evaluate_marcoqa(args.model_name, args.data_path, args.mode, args.output_dir, args.evaluate_result_dir)
