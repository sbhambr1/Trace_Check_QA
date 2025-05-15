import os
import ast
import json
from tqdm import tqdm
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_bleu_score(predicted_answer, gold_answer):
    """
    Compute BLEU score between predicted and gold answers.
    """
    # Tokenize the input strings
    predicted_tokens = predicted_answer.split()
    gold_tokens = [gold_answer.split()]  # Reference must be a list of token lists

    # Compute BLEU score with smoothing
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(gold_tokens, predicted_tokens, smoothing_function=smoothie)
    return bleu_score

def compute_rouge_score(predicted_answer, gold_answer):
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between predicted and gold answers.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gold_answer, predicted_answer)
    return scores
    

def get_gold_facts(gold_facts_text):
    facts = ast.literal_eval(gold_facts_text) if pd.notna(gold_facts_text) else []
    answers = ast.literal_eval(gold_facts_text) if pd.notna(gold_facts_text) else []
    answer_facts = [fact for fact in facts if any(answer in fact for answer in answers)]
    return answer_facts

def get_gold_category(gold_category_text):
    if 'during' in gold_category_text:
                gold_category = 'during'
    elif 'equal' in gold_category_text:
        gold_category = 'equal'
    elif 'overlap' in gold_category_text:
        gold_category = 'overlap'
    elif 'mix' in gold_category_text:
        gold_category = 'mix'
        
    return gold_category

def get_trace_eval(trace_eval_path, golden_eval_path):
    category_accuracy = 0.0
    fact_accuracy = 0.0
    bleu_score = 0.0
    rogue_score = 0.0
    trace_length = 0.0
    total = 0.0
    
    golden_csv = pd.read_csv(golden_eval_path)
    
    with open(trace_eval_path, 'r') as f:
        for line in tqdm(f):
            total += 1.0
            line = json.loads(line)
            if 'think' not in line['prediction']:
                continue
            predicted_answer = line['prediction']
            predicted_trace = predicted_answer.split('</think>')[0]
            if predicted_trace == '':
                continue
            predicted_category = predicted_trace.split('.')[0]
            if len(predicted_trace.split('.')) > 1:
                predicted_fact = predicted_trace.split('.')[1]
            else:
                predicted_fact = ''
            gold_answer = line['gold'][0]
            for index in range(len(golden_csv)):
                if gold_answer in golden_csv.iloc[index]['answer']:
                    row = golden_csv.iloc[index]
                    break
                
            gold_category = get_gold_category(row['reasoning'])
            gold_facts = get_gold_facts(row['facts'])
            
            facts_text = "I need to use the following facts to answer the question: " + str(gold_facts)
            reasoning_text = row['reasoning'] + "" + facts_text
            
            if gold_category in predicted_category:
                category_accuracy += 1.0
            for fact in gold_facts:
                if fact in predicted_trace:
                    fact_accuracy += 1.0
                    break
            bleu_score += compute_bleu_score(predicted_answer, reasoning_text)
            # rouge_scores['rouge1'] += compute_rouge_score(predicted_answer, reasoning_text)['rouge1'].fmeasure
            trace_length += len(predicted_trace.split())
            
            
    return category_accuracy/total, fact_accuracy/total, bleu_score/total, rogue_score/total, trace_length/total

def main(adapter_name):

    categories = ['equal', 'during', 'mix', 'overlap']
    
    total_accuracy = 0.0
    total_fact_accuracy = 0.0
    total_bleu_score = 0.0
    total_rogue_score = 0.0
    total_trace_length = 0.0

    for category in categories:
        
        trace_eval_path = f'results/Cotempqa/cotempqa_evaluation_outputs/{category}/{adapter_name}'
        if 'reasoning-facts' in adapter_name:
            golden_eval_path = f'data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/{category}_test.csv'
        elif 'reasoning' in adapter_name:
            golden_eval_path = f'data/cotempqa/sft_dataset_reasoning_chat_template/{category}_test.csv'
        else:
            golden_eval_path = f'data/cotempqa/sft_dataset_chat_template/{category}_test.csv'
            
        category_accuracy, fact_accuracy, bleu_score, rogue_score, avg_trace_length = get_trace_eval(trace_eval_path, golden_eval_path)
        
        total_accuracy += category_accuracy
        total_fact_accuracy += fact_accuracy
        total_bleu_score += bleu_score
        total_rogue_score += rogue_score
        total_trace_length += avg_trace_length
        
    avg_accuracy = total_accuracy / len(categories)
    avg_fact_accuracy = total_fact_accuracy / len(categories)
    avg_bleu_score = total_bleu_score / len(categories)
    avg_rogue_score = total_rogue_score / len(categories)
    avg_trace_length = total_trace_length / len(categories)

    print("Average Category Accuracy:", avg_accuracy*100)
    print("Average Fact Accuracy:", avg_fact_accuracy*100)
    print("Average BLEU Score:", avg_bleu_score)
    # print("Average ROUGE Score:", avg_rogue_score)
    print("Average Trace Length:", avg_trace_length)

if __name__ == "__main__":
    
    # adapter_name = 'Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts-perturbed_default.json'
    adapter_names = "Qwen3-1.7B-sft-adapter-reasoning-facts-perturbed_default.json" , "Qwen3-1.7B-sft-adapter-reasoning-facts_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts-perturbed_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts_default.json"
    
    for adapter_name in adapter_names:
        print('\n\n')
        print("="*50)
        print(f"Evaluating adapter: {adapter_name}") 