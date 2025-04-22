import os
import ast
import sys
import json
import subprocess
import pandas as pd
import networkx as nx
from tqdm import tqdm
from datasets import load_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.conversation import Conversation
from cat_bench_prompt import PromptGenerator

# llm_models: gpt-3.5-turbo, gpt-4o-mini, gpt-4o, llama3:8b, claude-3-haiku-20240307 (small), claude-3-sonnet-20240229 (medium), claude-3-opus-20240229 (large)
LLM_MODEL = "llama3:8b"
STORAGE_DIR = os.path.join(os.getcwd(), "storage/user_study")
EXPT_NAME = 'all_long_chain_questions'

class UserStudySetup():
    def __init__(self, dataset, result_csv):
        self.dataset = dataset
        self.result_csv = result_csv
        self.result_csv['steps'] = self.result_csv['steps'].apply(ast.literal_eval)
        self.total_dags = 0
        self.questionnaire = {
            "recipe_id": [],
            "origin_text": [],
            "destination_text": [],
            "steps": [],
            "path_length": []
        }
        self.llm_responses = {
            "recipe_id": [],
            "origin_text": [],
            "destination_text": [],
            "steps": [],
            "path_length": [],
            "llm_response": []
        }
        
        # debugging
        self.error_encountered = 0

    def get_true_edges(self, recipe_id):
        edges = []
        for k in range(len(self.result_csv)):
            if self.result_csv.iloc[k]['recipe_name'] == recipe_id and self.result_csv.iloc[k]['label'] == 1:
                steps = self.result_csv.iloc[k]['steps']
                enumerated_steps = list(enumerate(steps, start=1))
                sentence1 = self.result_csv.iloc[k]['sentence1']
                sentence2 = self.result_csv.iloc[k]['sentence2']
                i = next((step[0] for step in enumerated_steps if step[1] == sentence1), None)
                j = next((step[0] for step in enumerated_steps if step[1] == sentence2), None)
                edges.append([j, i])
                
        return edges
        
    def is_long_chain_question(self, recipe_id, i, j):
        G_true = nx.DiGraph()
        edges = self.get_true_edges(recipe_id)
        G_true.add_edges_from(edges)
        all_paths = nx.all_simple_paths(G_true, j, i)
        try:
            for path in all_paths:
                if len(path) > 2:
                    return True, len(path)
        except:
            self.error_encountered += 1
        return False, 0
        
    def get_long_chain_questions(self):
        for k in tqdm(range(len(self.dataset))):
            recipe_id, recipe_steps, i, j, label = PromptGenerator.format_input_for_prompt(self.dataset[k], temporal_relation='before')
            if label == 0:
                continue
            flag, path_length = self.is_long_chain_question(recipe_id, i, j)
            if flag:
                self.questionnaire["recipe_id"].append(recipe_id)
                self.questionnaire["origin_text"].append(i)
                self.questionnaire["destination_text"].append(j)
                self.questionnaire["steps"].append(recipe_steps)
                self.questionnaire["path_length"].append(path_length)
            
        sorted_questionnaire = sorted(zip(self.questionnaire["recipe_id"], self.questionnaire["origin_text"], self.questionnaire["destination_text"], self.questionnaire["steps"], self.questionnaire["path_length"]), key=lambda x: x[4])

        # Create a DataFrame from the sorted questionnaire
        df = pd.DataFrame(sorted_questionnaire, columns=["recipe_id", "origin_text", "destination_text", "steps", "path_length"])
        
        return df, self.questionnaire, self.error_encountered
    
    def get_long_chain_path(self, recipe_id, i, j):
        G_true = nx.DiGraph()
        edges = self.get_true_edges(recipe_id)
        G_true.add_edges_from(edges)
        all_paths = nx.all_simple_paths(G_true, j, i)
        try:
            for path in all_paths:
                if len(path) > 2:
                    return list(path)
        except:
            self.error_encountered += 1
        return None
    
    def get_templated_short_explanations(self, questions_csv):
        short_explanations = []
        for i in range(len(questions_csv)):
            origin_text_step = questions_csv.iloc[i]['origin_text']
            destination_text_step = questions_csv.iloc[i]['destination_text']
            templated_exp = "Step " + str(destination_text_step) + " is a prerequisite for Step " + str(origin_text_step) + "."
            short_explanations.append(templated_exp)
            
        return short_explanations
    
    def get_templated_long_explanations(self, questions_csv):
        long_explanations = []
        for i in range(len(questions_csv)):
            origin_text_step = questions_csv.iloc[i]['origin_text']
            destination_text_step = questions_csv.iloc[i]['destination_text']
            recipe_id = questions_csv.iloc[i]['recipe_id']
            path = self.get_long_chain_path(recipe_id, origin_text_step, destination_text_step)
            num_steps = len(path) - 2

            intermediate_steps = []
            for step in path[1:-1]:
                intermediate_steps.append("Step " + str(step))

            templated_exp = "Step " + str(destination_text_step) + " is a prerequisite for " + ", which is a prerequisite for ".join(intermediate_steps) + ", which is a prerequisite for Step " + str(origin_text_step) + "."
            long_explanations.append(templated_exp)
            
        return long_explanations
    
    def get_free_form_llm_explanations(self, llm_model):
        llm_explanations_list = []
        c = Conversation(llm_model, temp=0.5)
        for k in tqdm(range(len(self.dataset))):
            recipe_id, recipe_steps, i, j, label = PromptGenerator.format_input_for_prompt(self.dataset[k], temporal_relation='before')
            if label == 0:
                continue
            _, path_length = self.is_long_chain_question(recipe_id, i, j)
            origin_text = self.dataset[k]['origin_text']
            destination_text = self.dataset[k]['destination_text']
            prompt = f"Given the recipe steps:\n{recipe_steps}\n\nPlease provide a brief explanation in one line for why Step {j} is a prerequisite for Step {i}."
            response = c.get_response(prompt)
            llm_explanations_list.append(response)
            self.llm_responses["recipe_id"].append(recipe_id)
            self.llm_responses["origin_text"].append(origin_text)
            self.llm_responses["destination_text"].append(destination_text)
            self.llm_responses["steps"].append(recipe_steps)
            self.llm_responses["path_length"].append(path_length)
            self.llm_responses["llm_response"].append(response)
            
        llm_csv = pd.DataFrame(self.llm_responses, columns=["recipe_id", "origin_text", "destination_text", "steps", "path_length", "llm_response"])
        
        return self.llm_responses, llm_csv            
    
if __name__ == "__main__":
    
    # Load the dataset
    ds = load_dataset("vanyacohen/CaT-Bench")
    test_ds = ds['test']
    
    DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
    binary_label_path = os.path.join(DATASET_PATH, "binary_label")
    original_csv = pd.read_csv(os.path.join(binary_label_path, 'eval_test.csv'))
    
    user_study = UserStudySetup(test_ds, original_csv)
    
    csv_file = os.path.join(STORAGE_DIR, EXPT_NAME + ".csv")
    templated_exp_csv_file = os.path.join(STORAGE_DIR, EXPT_NAME + "_templated_explanations.csv")
    
    llm_model_responses_save_file = os.path.join(STORAGE_DIR, LLM_MODEL + "_" + EXPT_NAME + "_llm_responses.json")
    llm_model_responses_csv = os.path.join(STORAGE_DIR, LLM_MODEL + "_" + EXPT_NAME + "_llm_responses.csv")
    
    # Check if the user study questionnaire already exists
    if os.path.exists(csv_file):
        print('[INFO] User study questionnaire already exists:', csv_file)
        df = pd.read_csv(csv_file)            
    else:
        df, questionnaire, errors = user_study.get_long_chain_questions()
        
        print('[INFO] Total number of long chain questions:', len(questionnaire["recipe_id"]))
        
        if os.path.exists(STORAGE_DIR) == False:
            os.makedirs(STORAGE_DIR)
            
        save_file = os.path.join(STORAGE_DIR, EXPT_NAME+".json")
        with open(save_file, 'w') as f:
            json.dump(questionnaire, f)
        print('[INFO] User study questionnaire saved in:', save_file)
        
        # Save the DataFrame as a CSV file
        df.to_csv(csv_file, index=False)
        print('[INFO] User study questionnaire saved in:', csv_file)
    
    # Check if the templated explanations already exists    
    if os.path.exists(templated_exp_csv_file):
            print('[INFO] Templated explanations already exists:', templated_exp_csv_file)
    else:
        # Get templated explanations
        short_explanations = user_study.get_templated_short_explanations(df)
        long_explanations = user_study.get_templated_long_explanations(df)
        
        # Copy the DataFrame and add two extra columns for short and long explanations
        df_copy = df.copy()
        df_copy['short_explanation'] = short_explanations
        df_copy['long_explanation'] = long_explanations

        # Save the DataFrame as a CSV file
        df_copy.to_csv(templated_exp_csv_file, index=False)
        print('[INFO] Templated explanations saved in:', templated_exp_csv_file)
    
    # Check if the LLM responses already exists for the given model    
    if os.path.exists(llm_model_responses_save_file):
        print('[INFO] LLM responses already exists:', llm_model_responses_save_file)
    else:
        # Get LLM responses
        llm_responses_dict, llm_csv = user_study.get_free_form_llm_explanations(LLM_MODEL)
        
        # Save the responses in a JSON file
        with open(llm_model_responses_save_file, 'w') as f:
            json.dump(llm_responses_dict, f)
        print('[INFO] LLM responses saved in:', llm_model_responses_save_file)
        
        # Save the responses in a CSV file
        llm_csv.to_csv(llm_model_responses_csv, index=False)
        print('[INFO] LLM responses saved in:', llm_model_responses_csv)
    
        if "llama" in LLM_MODEL:
            command_stop = f"ollama stop {LLM_MODEL}"
            subprocess.run(command_stop, shell=True)