import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from dag_evaluation import EvaluateDAG
from cat_bench_prompt import PromptGenerator

STORAGE_DIR = os.path.join(os.getcwd(), "storage/dag_visualization")
EXPT_NAME = 'true_dag_visualizations'

save_path = os.path.join(STORAGE_DIR, EXPT_NAME)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
def main():
    
    ds = load_dataset("vanyacohen/CaT-Bench")
    
    test_ds = ds['test']
    
    # Load the dataset
    DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
    binary_label_path = os.path.join(DATASET_PATH, "binary_label")
    
    original_csv = pd.read_csv(os.path.join(binary_label_path, 'eval_test.csv'))
    
    # UPDATE TO MODEL'S PREDICTION TO GET MODEL PREDICTED DAGS
    result_csv = original_csv.copy()
    result_csv['prediction'] = result_csv['label']
    
    recipe_hashmap = {}
    evaluator = EvaluateDAG(result_csv)
    
    # Visualization
    for i in tqdm(range(len(test_ds))):
        recipe_id, recipe_steps, _, _, _ = PromptGenerator.format_input_for_prompt(test_ds[i], temporal_relation='before')
        
        if recipe_id not in recipe_hashmap:
                recipe_hashmap[recipe_id] = {}
        else:
            continue
        
        recipe_save_dir = os.path.join(save_path, str(recipe_id))
        if not os.path.exists(recipe_save_dir):
            os.makedirs(recipe_save_dir)
    
        true_dag, pred_dag, edges_true = evaluator.visualize_dags(recipe_id, save_path=recipe_save_dir)
        equivalent_dags = evaluator.are_equivalent(true_dag, pred_dag, edges_true)
        with open(os.path.join(recipe_save_dir, 'recipe_steps.txt'), 'w') as f:
            f.write(str(recipe_steps))
        with open(os.path.join(recipe_save_dir, 'equivalent_dags.txt'), 'w') as f:
            f.write(str(equivalent_dags))
            
    print('[INFO] DAG Visualizations saved in:', save_path)
    
if __name__ == "__main__":
    main()