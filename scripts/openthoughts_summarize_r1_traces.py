import os
import json
from tqdm import tqdm
from datasets import load_dataset
from conversation import Conversation

def create_summarization_prompt(trace):
    """Create a standard community-accepted summarization prompt template"""
    prompt = f"Summarize the following trace in a very concise and clear manner, highlighting key events and outcomes in less than 100 words:\n\n{trace}\n\nSummary:"
    return prompt

def summarize_trace(llm, trace):
    """Call API to summarize the trace"""
    prompt = create_summarization_prompt(trace)
    summary = llm.get_response(prompt)
    return summary

def main():
    
    summarizing_model = "gpt-4o-mini"
    llm = Conversation(summarizing_model, temp=0)
    
    """Main function to process the JSON file"""
    all_results_with_summary = []
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
    seed = 42
    ds.shuffle(seed=seed)
    data = ds.filter(lambda x: x['domain'] == 'math')[:3000]
    
    for i in tqdm(range(3000)):
        problem = data['problem'][i]
        deepseek_trace = data['deepseek_reasoning'][i]
        deepseek_solution = data['deepseek_solution'][i]
        ground_truth_solution = data['ground_truth_solution'][i]
        domain = data['domain'][i]
        source = data['source'][i]
        summary = summarize_trace(llm, deepseek_trace)
        new_data = {
            'input': problem,
            'prediction': deepseek_solution,
            'ground_truth_solution': ground_truth_solution,
            'r1_trace': deepseek_trace,
            'r1_trace_summary': summary,
            'domain': domain,
            'source': source
        }
        all_results_with_summary.append(new_data)
        
            
    save_path = 'results/OpenThoughts/deepseek_r1_with_summary_' + summarizing_model + '.json'
    with open(save_path, 'w') as f:
        for record in all_results_with_summary:
            f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    
    main()
