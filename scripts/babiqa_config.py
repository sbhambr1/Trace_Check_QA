import argparse
import os
import itertools
import json
import jsonlines
import datetime
import string
from collections import Counter
import openai
import re

default_template = 'Answer the question based on the context:\n{fact}\nQuestion: {question}\nOnly return the answer.\n'

default_template_with_trace = '''Answer the question based on the context:\n{fact}\nQuestion: {question}\n<think> {reasoning_trace} </think>\n Only return the answer.\n'''

def get_prompts(all_inputs, template):
    """
    Generate prompts from the input data using the provided template.
    
    Parameters:
    all_inputs (list): List of input data dictionaries.
    template (str): Template string for formatting the prompts.
    
    Returns:
    list: List of formatted prompts.
    """
    all_outputs = []
    for input in all_inputs:
        fact_str = "\n".join(input['facts'])
        output = template.format(
            fact=fact_str,
            question=input['question']
        )
        all_outputs.append(output)
    return all_outputs

def get_prompts_with_trace(all_inputs, template):
    """
    Generate prompts from the input data using the provided template.
    
    Parameters:
    all_inputs (list): List of input data dictionaries.
    template (str): Template string for formatting the prompts.
    
    Returns:
    list: List of formatted prompts.
    """
    all_outputs = []
    for input in all_inputs:
        fact_str = "\n".join(input['facts'])
        output = template.format(
            fact=fact_str,
            question=input['question'],
            reasoning_trace=input['reasoning_trace']
        )
        all_outputs.append(output)
    return all_outputs


def chatgpt(out_f, prompt, item_data, index, api_list):
    """
    Interact with the OpenAI ChatGPT API to get a response for the given prompt.
    
    Parameters:
    out_f (file object): File object to write the output.
    prompt (str): Prompt to send to the API.
    item_data (dict): Input data dictionary.
    index (int): Index to select the API key from the list.
    api_list (list): List of OpenAI API keys.
    """
    cnt = 0
    key_index = index % len(api_list)
    while cnt < 5:
        try:
            openai.api_key = api_list[key_index]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            item_data['ans'] = answer
            json_data = json.dumps(item_data)
            out_f.write(json_data + '\n')
            out_f.flush()
            break
        except Exception as e:
            print(e)
            cnt += 1
            continue

def normalize_answer(s):
    """
    Normalize the text by converting to lowercase, removing punctuation, and fixing whitespace.
    
    Parameters:
    s (str): Input string to normalize.
    
    Returns:
    str: Normalized string.
    """
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match_score(prediction, ground_truth):
    """
    Compute the exact match score between the prediction and ground truth.
    
    Parameters:
    prediction (str): Predicted answer string.
    ground_truth (str): Ground truth answer string.
    
    Returns:
    bool: True if prediction matches ground_truth exactly, else False.
    """
    return prediction == ground_truth

def f1_score(prediction, ground_truth):
    """
    Compute the F1 score, precision, and recall between the prediction and ground truth.
    
    Parameters:
    prediction (str): Predicted answer string.
    ground_truth (str): Ground truth answer string.
    
    Returns:
    tuple: F1 score, precision, and recall.
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return (0, 0, 0)
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


def evaluate_model(all_data, mode):
    """
    Evaluate the performance of predictions against the ground truth.
    
    Parameters:
    all_data (list): List of data dictionaries with predictions and ground truth.
    mode (str): Evaluation mode (e.g., 'cot' for chain of thought).
    
    Returns:
    dict: Evaluation metrics including accuracy, F1 score, precision, recall, and average score.
    """
    em_total = 0
    f1_total = 0
    p_total = 0
    r_total = 0
    count = 0
    
    for data in all_data:
        golds = data['answer']
        golds = [ans.lower() for ans in golds]

        prediction = data['prediction'].lower()
        if 'cot' in mode:
            if 'therefore the answer is' not in prediction:
                prediction = 'answer'
            else:
                prediction = prediction.split('therefore the answer is')[1].split('answer the question based on')[0]
        elif 'answer the question based on' in prediction:
            prediction = prediction.split('answer the question based on')[0]
        elif ' answer ' in prediction:
            prediction = prediction.split(' answer ')[1]
        # Adding for SFT models, need to check for other models in prompting
        elif 'answer:' in prediction:
            prediction = prediction.split('answer:')[1]
        elif '<answer>' in prediction:
            prediction = prediction.split('<answer>')[1]

        question = data['question'].lower()

        shot = False

        predict = []

        ans = golds
        if ans in prediction:
            prediction = prediction.replace(ans, '')
            shot = True

        if shot:
            predict = [prediction]

        predict = list(set(predict))
        predict = [normalize_answer(i) for i in predict]
        predict.sort()
        golds = list(set(golds))
        golds = [normalize_answer(i) for i in golds]
        golds.sort()

        em_total += exact_match_score(predict, golds)
        f1, p, r = f1_score(predict, golds)
        f1_total += f1
        p_total += p
        r_total += r
        count += 1

    return {
        'acc': round(em_total * 100 / count, 1),
        'f1': round(f1_total * 100 / count, 1),
        'p': round(p_total * 100 / count, 1),
        'r': round(r_total * 100 / count, 1),
        'avg': round((em_total + f1_total) * 50 / count, 1)
    }




