import os
import json
import pandas as pd
import os
import matplotlib.pyplot as plt

def visualize(modes, reasoning=False):
    df = pd.DataFrame(columns=['model', 'mode', 'accuracy', 'f1', 'precision', 'recall', 'avg'])
    
    model_ids = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "gemma-3-1b-it", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Llama-8B"]

    for mode in modes:
        evaluate_result_dir = f"results/babiqa/evaluation_results/{mode}/"
        
        if not os.path.exists(evaluate_result_dir):
            print(f"Directory {evaluate_result_dir} does not exist.")
            continue
        
        for file_name in os.listdir(evaluate_result_dir):
            if file_name.endswith(".json"):
                for model_name in model_ids:
                    if model_name in file_name:
                        with open(os.path.join(evaluate_result_dir, file_name), 'r') as f:
                            data = json.load(f)
                            accuracy = data.get('acc')
                            f1 = data.get('f1')
                            precision = data.get('p')
                            recall = data.get('r')
                            avg = data.get('avg')
                            
                            new_row = {
                                'model': model_name,
                                'mode': mode,
                                'accuracy': accuracy,
                                'f1': f1,
                                'precision': precision,
                                'recall': recall,
                                'avg': avg
                            }
                            
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)    
                        
    df.sort_values(by=['model', 'mode', 'data_type'], inplace=True)
    
    # Visualize accuracy for each model
    models = df['model'].unique()
    
    reshaped_df = df.pivot_table(
        index='model', 
        columns=['mode'], 
        values=['accuracy', 'f1', 'precision', 'recall', 'avg']
    )
    
    visualization_path = "visualization/"
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    if reasoning:
        reshaped_df.to_csv('visualization/reshaped_df_with_reasoning.csv', index=True)
    else:
        reshaped_df.to_csv('visualization/reshaped_df.csv', index=True)

    return reshaped_df  
    
        
    
if __name__ == '__main__':
    
    modes=["default"]
    visualize(modes)