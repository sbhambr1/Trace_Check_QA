import os
import sys
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ds = load_dataset("vanyacohen/CaT-Bench")
# Note: destination idx < origin idx for temporal relation "before"

STORAGE_DIR = os.path.join(os.getcwd(), "data_storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

def convert_to_df_binary(ds):
    train_ds, test_ds, valid_ds = ds['train'], ds['test'], ds['validation']
    df_binary_train = pd.DataFrame(columns=['index', 'recipe_name', 'steps', 'sentence1', 'sentence2', 'label'])
    df_binary_test = pd.DataFrame(columns=['index', 'recipe_name', 'steps', 'sentence1', 'sentence2', 'label'])
    df_binary_valid = pd.DataFrame(columns=['index', 'recipe_name', 'steps', 'sentence1', 'sentence2', 'label'])
    
    print('Processing conversion to binary-label dataframe')
    for i, data in tqdm(enumerate(train_ds)):
        if data['label'] == True:
            df_binary_train.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 1]
        else:
            df_binary_train.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 0]
    
    for i, data in tqdm(enumerate(test_ds)):
        if data['label'] == True:
            df_binary_test.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 1]
        else:
            df_binary_test.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 0]
            
    for i, data in tqdm(enumerate(valid_ds)):
        if data['label'] == True:
            df_binary_valid.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 1]
        else:
            df_binary_valid.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 0]
            
    df_binary_eval_train = df_binary_train.copy()
    df_binary_eval_test = df_binary_test.copy()
    df_binary_eval_valid = df_binary_valid.copy()
    df_binary_train.drop(columns=['steps'], inplace=True)
    df_binary_test.drop(columns=['steps'], inplace=True)
    df_binary_valid.drop(columns=['steps'], inplace=True)
            
    return df_binary_train, df_binary_test, df_binary_valid, df_binary_eval_train, df_binary_eval_test, df_binary_eval_valid
    
def convert_to_df_multi(ds):
    train_ds, test_ds, valid_ds = ds['train'], ds['test'], ds['validation']
    df_multi_train = pd.DataFrame(columns=['index', 'recipe_name', 'steps', 'sentence1', 'sentence2', 'label'])
    df_multi_test = pd.DataFrame(columns=['index', 'recipe_name', 'steps', 'sentence1', 'sentence2', 'label'])
    df_multi_valid = pd.DataFrame(columns=['index', 'recipe_name', 'steps', 'sentence1', 'sentence2', 'label'])
    
    print('Processing conversion to multi-label dataframe')
    for i, data in tqdm(enumerate(train_ds)):
        if data['label'] == True:
            df_multi_train.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 1]
        else:
            df_multi_train.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 0]
            
    for i, data in tqdm(enumerate(train_ds)):
        new_idx = i + len(train_ds)
        if data['label'] == True:
            df_multi_train.loc[new_idx] = [new_idx, data['title'], data['steps'], data['destination_text'], data['origin_text'], 2]
            
    for i, data in tqdm(enumerate(test_ds)):
        if data['label'] == True:
            df_multi_test.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 1]
        else:
            df_multi_test.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 0]
            
    for i, data in tqdm(enumerate(test_ds)):
        new_idx = i + len(test_ds)
        if data['label'] == True:
            df_multi_test.loc[new_idx] = [new_idx, data['title'], data['steps'], data['destination_text'], data['origin_text'], 2]
            
    for i, data in tqdm(enumerate(valid_ds)):
        if data['label'] == True:
            df_multi_valid.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 1]
        else:
            df_multi_valid.loc[i] = [i, data['title'], data['steps'], data['origin_text'], data['destination_text'], 0]
            
    for i, data in tqdm(enumerate(valid_ds)):
        new_idx = i + len(valid_ds)
        if data['label'] == True:
            df_multi_valid.loc[new_idx] = [new_idx, data['title'], data['steps'], data['destination_text'], data['origin_text'], 2]  
            
    df_multi_eval_train = df_multi_train.copy()
    df_multi_eval_test = df_multi_test.copy()
    df_multi_eval_valid = df_multi_valid.copy()
    df_multi_train.drop(columns=['steps'], inplace=True)
    df_multi_test.drop(columns=['steps'], inplace=True)
    df_multi_valid.drop(columns=['steps'], inplace=True)
            
    return df_multi_train, df_multi_test, df_multi_valid, df_multi_eval_train, df_multi_eval_test, df_multi_eval_valid

def main():
    df_multi_train, df_multi_test, df_multi_valid, df_multi_eval_train, df_multi_eval_test, df_multi_eval_valid = convert_to_df_multi(ds)
    df_binary_train, df_binary_test, df_binary_valid, df_binary_eval_train, df_binary_eval_test, df_binary_eval_valid = convert_to_df_binary(ds)   
    
    data_storage_path = os.path.join(STORAGE_DIR, 'cat_bench')
    if not os.path.exists(data_storage_path):
        os.makedirs(data_storage_path)
        
    binary_label_path = os.path.join(data_storage_path, 'binary_label')
    multi_label_path = os.path.join(data_storage_path, 'multi_label')
    
    if not os.path.exists(binary_label_path):
        os.makedirs(binary_label_path)
        
    if not os.path.exists(multi_label_path):
        os.makedirs(multi_label_path)
        
    df_binary_train.to_csv(os.path.join(binary_label_path, 'train.csv'), index=False)
    df_binary_test.to_csv(os.path.join(binary_label_path, 'test.csv'), index=False)
    df_binary_valid.to_csv(os.path.join(binary_label_path, 'valid.csv'), index=False)
    df_binary_eval_train.to_csv(os.path.join(binary_label_path, 'eval_train.csv'), index=False)
    df_binary_eval_test.to_csv(os.path.join(binary_label_path, 'eval_test.csv'), index=False)
    df_binary_eval_valid.to_csv(os.path.join(binary_label_path, 'eval_valid.csv'), index=False)
    
    df_multi_train.to_csv(os.path.join(multi_label_path, 'train.csv'), index=False)
    df_multi_test.to_csv(os.path.join(multi_label_path, 'test.csv'), index=False)
    df_multi_valid.to_csv(os.path.join(multi_label_path, 'valid.csv'), index=False)
    df_multi_eval_train.to_csv(os.path.join(multi_label_path, 'eval_train.csv'), index=False)
    df_multi_eval_test.to_csv(os.path.join(multi_label_path, 'eval_test.csv'), index=False)
    df_multi_eval_valid.to_csv(os.path.join(multi_label_path, 'eval_valid.csv'), index=False)
    
    print('Dataframes saved successfully at:', data_storage_path)
    
if __name__ == "__main__":
    main()