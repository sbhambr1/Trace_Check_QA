import os
import ast
import json
from tqdm import tqdm
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from cotempqa_config import *

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

def get_confusion_matrices(trace_eval_path, golden_eval_path):
    category_accuracy = 0.0
    fact_accuracy = 0.0
    answer_accuracy = 0.0
    total = 0.0
    unevaluated = 0
    
    # matrix 1
    correct_answer_correct_category = 0.0
    correct_answer_incorrect_category = 0.0
    incorrect_answer_correct_category = 0.0
    incorrect_answer_incorrect_category = 0.0
    
    # matrix 2
    correct_answer_correct_fact = 0.0
    correct_answer_incorrect_fact = 0.0
    incorrect_answer_correct_fact = 0.0
    incorrect_answer_incorrect_fact = 0.0
    
    # matrix 3
    correct_answer_correct_trace = 0.0
    correct_answer_incorrect_trace = 0.0
    incorrect_answer_correct_trace = 0.0
    incorrect_answer_incorrect_trace = 0.0
    
    golden_csv = pd.read_csv(golden_eval_path)
    
    mode = 'default'
    
    with open(trace_eval_path, 'r') as f:
        for line in tqdm(f):
            total += 1.0
            line = json.loads(line)
            
            result = evaluate_model([line], mode)
            correct_answer = 1 if result['acc'] == 100.0 else 0
            answer_accuracy += correct_answer
            
            if 'think' not in line['prediction']:
                unevaluated +=1
                continue
            predicted_answer = line['prediction']
            predicted_trace = predicted_answer.split('</think>')[0]
            if predicted_trace == '':
                unevaluated +=1
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
                correct_category = 1
                if correct_answer == 1:
                    correct_answer_correct_category += 1.0
                else:
                    incorrect_answer_correct_category += 1.0
            else:
                correct_category = 0
                if correct_answer == 1:
                    correct_answer_incorrect_category += 1.0
                else:
                    incorrect_answer_incorrect_category += 1.0
            
            flag_fact = 0
            for fact in gold_facts:
                if fact in predicted_trace:
                    correct_fact = 1
                    fact_accuracy += 1.0
                    if correct_answer == 1:
                        correct_answer_correct_fact += 1.0
                    else:
                        incorrect_answer_correct_fact += 1.0
                    flag_fact = 1
                    break
            if not flag_fact:
                correct_fact = 0
                if correct_answer == 1:
                    correct_answer_incorrect_fact += 1.0
                else:
                    incorrect_answer_incorrect_fact += 1.0

                        
            got_added = 0
            if correct_category == 1 and correct_fact == 1:
                if correct_answer == 1:
                    correct_answer_correct_trace += 1.0
                    got_added = 1
                else:
                    incorrect_answer_correct_trace += 1.0
                    got_added = 1
            else:
                if correct_answer == 1:
                    correct_answer_incorrect_trace += 1.0
                    got_added = 1
                else:
                    incorrect_answer_incorrect_trace += 1.0
                    got_added = 1
                    
                    
            checking = None
                
                
    
    accuracies = {
        'unevaluated': unevaluated,
        'category_accuracy': category_accuracy / (total),
        'fact_accuracy': fact_accuracy / (total),
        'answer_accuracy': answer_accuracy / (total)
    }
    
    matrix1 = {
        'correct_answer_correct_category': correct_answer_correct_category / (total-unevaluated),
        'correct_answer_incorrect_category': correct_answer_incorrect_category / (total-unevaluated),
        'incorrect_answer_correct_category': incorrect_answer_correct_category / (total-unevaluated),
        'incorrect_answer_incorrect_category': incorrect_answer_incorrect_category / (total-unevaluated)
    }
    matrix2 = {
        'correct_answer_correct_fact': correct_answer_correct_fact / (total-unevaluated),
        'correct_answer_incorrect_fact': correct_answer_incorrect_fact / (total-unevaluated),
        'incorrect_answer_correct_fact': incorrect_answer_correct_fact / (total-unevaluated),
        'incorrect_answer_incorrect_fact': incorrect_answer_incorrect_fact / (total-unevaluated),
    }
    matrix3 = {
        'correct_answer_correct_trace': correct_answer_correct_trace / (total-unevaluated),
        'correct_answer_incorrect_trace': correct_answer_incorrect_trace / (total-unevaluated),
        'incorrect_answer_correct_trace': incorrect_answer_correct_trace / (total-unevaluated),
        'incorrect_answer_incorrect_trace': incorrect_answer_incorrect_trace / (total-unevaluated),
    }
            
            
    return accuracies, matrix1, matrix2, matrix3

def main(adapter_name):

    categories = ['equal', 'during', 'mix', 'overlap']
    
    total_accuracies = {
        'category_accuracy': 0.0,
        'fact_accuracy': 0.0,
        'answer_accuracy': 0.0
    }
    total_matrix1 = {
        'correct_answer_correct_category': 0.0,
        'correct_answer_incorrect_category': 0.0,
        'incorrect_answer_correct_category': 0.0,
        'incorrect_answer_incorrect_category': 0.0
    }
    total_matrix2 = {
        'correct_answer_correct_fact': 0.0,
        'correct_answer_incorrect_fact': 0.0,
        'incorrect_answer_correct_fact': 0.0,
        'incorrect_answer_incorrect_fact': 0.0
    }
    
    total_matrix3 = {
        'correct_answer_correct_trace': 0.0,
        'correct_answer_incorrect_trace': 0.0,
        'incorrect_answer_correct_trace': 0.0,
        'incorrect_answer_incorrect_trace': 0.0
    }

    for category in categories:
        
        trace_eval_path = f'results/Cotempqa/evaluation_outputs/{category}/{adapter_name}'
        if 'reasoning-facts' in adapter_name:
            golden_eval_path = f'data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/{category}_test.csv'
        elif 'reasoning' in adapter_name:
            golden_eval_path = f'data/cotempqa/sft_dataset_reasoning_chat_template/{category}_test.csv'
        else:
            golden_eval_path = f'data/cotempqa/sft_dataset_chat_template/{category}_test.csv'
            
        category_accuracies, category_matrix1, category_matrix2, category_matrix3 = get_confusion_matrices(trace_eval_path, golden_eval_path)
        
        total_accuracies['category_accuracy'] += category_accuracies['category_accuracy']
        total_accuracies['fact_accuracy'] += category_accuracies['fact_accuracy']
        total_accuracies['answer_accuracy'] += category_accuracies['answer_accuracy']
        
        total_matrix1['correct_answer_correct_category'] += category_matrix1['correct_answer_correct_category']
        total_matrix1['correct_answer_incorrect_category'] += category_matrix1['correct_answer_incorrect_category']
        total_matrix1['incorrect_answer_correct_category'] += category_matrix1['incorrect_answer_correct_category']
        total_matrix1['incorrect_answer_incorrect_category'] += category_matrix1['incorrect_answer_incorrect_category']
        
        total_matrix2['correct_answer_correct_fact'] += category_matrix2['correct_answer_correct_fact']
        total_matrix2['correct_answer_incorrect_fact'] += category_matrix2['correct_answer_incorrect_fact']
        total_matrix2['incorrect_answer_correct_fact'] += category_matrix2['incorrect_answer_correct_fact']
        total_matrix2['incorrect_answer_incorrect_fact'] += category_matrix2['incorrect_answer_incorrect_fact']
        
        total_matrix3['correct_answer_correct_trace'] += category_matrix3['correct_answer_correct_trace']
        total_matrix3['correct_answer_incorrect_trace'] += category_matrix3['correct_answer_incorrect_trace']
        total_matrix3['incorrect_answer_correct_trace'] += category_matrix3['incorrect_answer_correct_trace']
        total_matrix3['incorrect_answer_incorrect_trace'] += category_matrix3['incorrect_answer_incorrect_trace']
        
    # Calculate the average scores
    total_accuracies['category_accuracy'] = round((total_accuracies['category_accuracy'] / len(categories))*100, 2)
    total_accuracies['fact_accuracy'] = round((total_accuracies['fact_accuracy'] / len(categories))*100, 2)
    total_accuracies['answer_accuracy'] = round((total_accuracies['answer_accuracy'] / len(categories))*100, 2)
    
    total_matrix1['correct_answer_correct_category'] = round((total_matrix1['correct_answer_correct_category'] / len(categories))*100, 2)
    total_matrix1['correct_answer_incorrect_category'] = round((total_matrix1['correct_answer_incorrect_category'] / len(categories))*100, 2)
    total_matrix1['incorrect_answer_correct_category'] = round((total_matrix1['incorrect_answer_correct_category'] / len(categories))*100, 2)
    total_matrix1['incorrect_answer_incorrect_category'] = round((total_matrix1['incorrect_answer_incorrect_category'] / len(categories))*100, 2)
    
    total_matrix2['correct_answer_correct_fact'] = round((total_matrix2['correct_answer_correct_fact'] / len(categories))*100, 2)
    total_matrix2['correct_answer_incorrect_fact'] = round((total_matrix2['correct_answer_incorrect_fact'] / len(categories))*100, 2)
    total_matrix2['incorrect_answer_correct_fact'] = round((total_matrix2['incorrect_answer_correct_fact'] / len(categories))*100, 2)
    total_matrix2['incorrect_answer_incorrect_fact'] = round((total_matrix2['incorrect_answer_incorrect_fact'] / len(categories))*100, 2)
    
    total_matrix3['correct_answer_correct_trace'] = round((total_matrix3['correct_answer_correct_trace'] / len(categories))*100, 2)
    total_matrix3['correct_answer_incorrect_trace'] = round((total_matrix3['correct_answer_incorrect_trace'] / len(categories))*100, 2)
    total_matrix3['incorrect_answer_correct_trace'] = round((total_matrix3['incorrect_answer_correct_trace'] / len(categories))*100, 2)
    total_matrix3['incorrect_answer_incorrect_trace'] = round((total_matrix3['incorrect_answer_incorrect_trace'] / len(categories))*100, 2)
    
    print('Accuracies:')
    print(total_accuracies)
    print('\n')
    
    print("Confusion Matrix 1:")
    print(total_matrix1)
    print('\n')
    
    print("Confusion Matrix 2:")
    print(total_matrix2)
    print('\n')
    
    print("Confusion Matrix 3:")
    print(total_matrix3)
    print('\n')


if __name__ == "__main__":
    
    # adapter_name = 'Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts-perturbed_default.json'
    adapter_names = "Qwen3-1.7B-sft-adapter-reasoning-facts-perturbed_default.json" , "Qwen3-1.7B-sft-adapter-reasoning-facts_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts-perturbed_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts_default.json"
    
    for adapter_name in adapter_names:
        print('\n\n')
        print("="*50)
        print(f"Evaluating adapter: {adapter_name}")
        main(adapter_name)