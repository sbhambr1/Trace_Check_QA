import os
import json
import pandas as pd
import os
import matplotlib.pyplot as plt

def visualize(data_types, modes, reasoning=False):
    df = pd.DataFrame(columns=['model', 'mode', 'data_type', 'accuracy', 'f1', 'precision', 'recall', 'avg'])
    
    model_ids = ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "gemma-3-1b-it", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Llama-8B"]

    for data_type in data_types:
        for mode in modes:
            evaluate_result_dir = f"results/Cotempqa/evaluation_results/{data_type}_{mode}/"
            
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
                                    'data_type': data_type,
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
        columns=['data_type', 'mode'], 
        values=['accuracy', 'f1', 'precision', 'recall', 'avg']
    )
    
    visualization_path = "visualization/"
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)
        
    # metrics = ['accuracy', 'f1', 'precision', 'recall', 'avg']

    # for data_type in reshaped_df['data_type'].unique():
    #     for mode in reshaped_df['mode'].unique():
    #         performance = reshaped_df[(df['data_type'] == data_type) & (reshaped_df['mode'] == mode)]['accuracy']
    #         plt.bar(models, performance)
    #         plt.xlabel('Model')
    #         plt.ylabel('Accuracy')
    #         plt.title(f'Accuracy for Each Model - Data Type: {data_type}, Mode: {mode}')
    #         plt.savefig(os.path.join(visualization_path, f'accuracy_{data_type}_{mode}.png'))

    if reasoning:
        reshaped_df.to_csv('visualization/reshaped_df_with_reasoning.csv', index=True)
    else:
        reshaped_df.to_csv('visualization/reshaped_df.csv', index=True)

    return reshaped_df  
    
        
    
if __name__ == '__main__':
    data_types=["mix", "equal", "during", "overlap"]
    modes=["default", "few_shot_cot", "few_shot"]

    visualize(data_types, modes)

    data_types=["mix_with_reasoning", "equal_with_reasoning", "during_with_reasoning", "overlap_with_reasoning"]
    modes=["default_with_reasoning"]
    
    visualize(data_types, modes, reasoning=True)