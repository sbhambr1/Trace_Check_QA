import os
import json
from tqdm import tqdm
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
    
    llm = Conversation("gpt-4o-mini", temp=0)
    
    """Main function to process the JSON file"""
    categories = ['during', 'equal', 'mix', 'overlap']
    all_results_with_summary = []
    for category in categories:
        print(f"Processing category: {category}")
        cat_results_with_summary = []
        cat_correct_responses_file = 'results/Cotempqa/deepseek_r1_correct_responses_' + category + '.json'
        with open(cat_correct_responses_file, 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                trace = data['r1_trace']
                if trace:
                    summary = summarize_trace(llm, trace)
                    data['r1_trace_summary'] = summary
                cat_results_with_summary.append(data)
                all_results_with_summary.append(data)
                
        cat_correct_responses_file_with_summary = 'results/Cotempqa/deepseek_r1_correct_responses_' + category + '_with_summary.json'
        with open(cat_correct_responses_file_with_summary, 'w') as f:
            for record in cat_results_with_summary:
                f.write(json.dumps(record) + '\n')
                
    correct_responses_file_with_summary = 'results/Cotempqa/deepseek_r1_correct_responses_with_summary.json'
    with open(correct_responses_file_with_summary, 'w') as f:
        for record in all_results_with_summary:
            f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    
    main()
