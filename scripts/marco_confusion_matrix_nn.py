import os
import ast
import json
from tqdm import tqdm
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from marco_config import *

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
    if 'description' in gold_category_text.lower():
        gold_category = 'description'
    elif 'numeric' in gold_category_text.lower():
        gold_category = 'numeric'
    elif 'entity' in gold_category_text.lower():
        gold_category = 'entity'
    elif 'location' in gold_category_text.lower():
        gold_category = 'location'
    elif 'person' in gold_category_text.lower():
        gold_category ='person'
        
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
            predicted_category = predicted_trace.split('.')[1] if len(predicted_trace.split('.')) > 1 else predicted_trace.split('.')[0]
            if len(predicted_trace.split('.')) > 1:
                predicted_fact = predicted_trace.split('.')[1]
            else:
                predicted_fact = ''
            gold_answer = line['gold']
            for index in range(len(golden_csv)):
                if gold_answer in golden_csv.iloc[index]['answers']:
                    row = golden_csv.iloc[index]
                    break
                
            gold_category = get_gold_category(row['reasoning'])
            gold_facts = get_gold_facts(row['passages'])
            
            facts_text = "I need to use the following facts to answer the question: " + str(gold_facts)
            reasoning_text = row['reasoning'] + "" + facts_text
            
            
            if gold_category in predicted_category.lower():
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
        'category_accuracy': category_accuracy,
        'fact_accuracy': fact_accuracy,
        'answer_accuracy': answer_accuracy
    }
    
    matrix1 = {
        'correct_answer_correct_category': correct_answer_correct_category,
        'correct_answer_incorrect_category': correct_answer_incorrect_category,
        'incorrect_answer_correct_category': incorrect_answer_correct_category,
        'incorrect_answer_incorrect_category': incorrect_answer_incorrect_category,
    }
    matrix2 = {
        'correct_answer_correct_fact': correct_answer_correct_fact,
        'correct_answer_incorrect_fact': correct_answer_incorrect_fact,
        'incorrect_answer_correct_fact': incorrect_answer_correct_fact,
        'incorrect_answer_incorrect_fact': incorrect_answer_incorrect_fact,
    }
    matrix3 = {
        'correct_answer_correct_trace': correct_answer_correct_trace,
        'correct_answer_incorrect_trace': correct_answer_incorrect_trace,
        'incorrect_answer_correct_trace': incorrect_answer_correct_trace,
        'incorrect_answer_incorrect_trace': incorrect_answer_incorrect_trace,
    }
            
            
    return accuracies, matrix1, matrix2, matrix3

def main(adapter_name):
        
    trace_eval_path = f'results/Marcoqa/evaluation_outputs/{adapter_name}'
    if 'reasoning-facts' in adapter_name:
        golden_eval_path = f'data/marcoqa/sft_dataset_reasoning_with_facts_chat_template/test.csv'
    elif 'reasoning' in adapter_name:
        golden_eval_path = f'data/marcoqa/sft_dataset_reasoning_chat_template/test.csv'
    else:
        golden_eval_path = f'data/marcoqa/sft_dataset_chat_template/test.csv'
        
    category_accuracies, category_matrix1, category_matrix2, category_matrix3 = get_confusion_matrices(trace_eval_path, golden_eval_path)
    
    return category_accuracies, category_matrix1, category_matrix2, category_matrix3


if __name__ == "__main__":
    
    adapter_names = "Qwen3-1.7B-sft-adapter-reasoning-facts-perturbed_default.json" , "Qwen3-1.7B-sft-adapter-reasoning-facts_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts-perturbed_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts_default.json"
    
    # adapter_names = "Qwen3-1.7B-sft-adapter-reasoning-facts_default.json", "Llama-3.2-1B-Instruct-sft-adapter-reasoning-facts_default.json"
    
    for adapter_name in adapter_names:
        print('\n\n')
        print("="*50)
        print(f"Evaluating adapter: {adapter_name}")
        total_accuracies, total_matrix1, total_matrix2, total_matrix3 = main(adapter_name)
        
        output_dir = 'results/Marcoqa/confusion_matrix/'
        save_name = adapter_name.split('.json')[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(os.path.join(output_dir, f'{save_name}_confusion_matrix.json'), 'w') as f:
            json.dump({
                'accuracies': total_accuracies,
                'matrix1': total_matrix1,
                'matrix2': total_matrix2,
                'matrix3': total_matrix3
            }, f, indent=4)
            
        print(f"Confusion matrix saved to {output_dir}{save_name}_confusion_matrix.json")
    