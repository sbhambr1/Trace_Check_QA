
import os
import sys
import warnings
import pandas as pd
from datasets import Dataset, load_dataset

warnings.filterwarnings("ignore")

ds_qa1 = load_dataset("facebook/babi_qa", type="en", task_no="qa1")
ds_qa2 = load_dataset("facebook/babi_qa", type="en", task_no="qa2")
ds_qa4 = load_dataset("facebook/babi_qa", type="en", task_no="qa4")
ds_qa7 = load_dataset("facebook/babi_qa", type="en", task_no="qa7")
ds_qa8 = load_dataset("facebook/babi_qa", type="en", task_no="qa8")
ds_qa12 = load_dataset("facebook/babi_qa", type="en", task_no="qa12")
ds_qa14 = load_dataset("facebook/babi_qa", type="en", task_no="qa14")
ds_qa15 = load_dataset("facebook/babi_qa", type="en", task_no="qa15") 
ds_qa16 = load_dataset("facebook/babi_qa", type="en", task_no="qa16")
ds_qa17 = load_dataset("facebook/babi_qa", type="en", task_no="qa17")
ds_qa18 = load_dataset("facebook/babi_qa", type="en", task_no="qa18")

print("Loading dataset...")
# print(ds_qa1)

# Loading dataset...
# DatasetDict({
#     train: Dataset({
#         features: ['story'],
#         num_rows: 2000
#     })
#     test: Dataset({
#         features: ['story'],
#         num_rows: 200
#     })
# })

# Get 1000 samples from the training set of each ds and 100 from the test set and combine inTo a single train and test set
train_dfs = []
test_dfs = []
for ds in [ds_qa1, ds_qa2, ds_qa4, ds_qa7, ds_qa8, ds_qa12, ds_qa14, ds_qa15, ds_qa16, ds_qa17, ds_qa18]:
    train_df = pd.DataFrame(ds['train'])
    test_df = pd.DataFrame(ds['test'])
    if ds == ds_qa1:
        train_df['qa'] = 'qa1'
        test_df['qa'] = 'qa1'  # Add a column indicating the QA dataset
    elif ds == ds_qa2:
        train_df['qa'] = 'qa2'
        test_df['qa'] = 'qa2'
    elif ds == ds_qa4:
        train_df['qa'] = 'qa4'
        test_df['qa'] = 'qa4'
    elif ds == ds_qa7:
        train_df['qa'] = 'qa7'
        test_df['qa'] = 'qa7'
    elif ds == ds_qa8:
        train_df['qa'] = 'qa8'
        test_df['qa'] = 'qa8'
    elif ds == ds_qa12:
        train_df['qa'] = 'qa12'
        test_df['qa'] = 'qa12'
    elif ds == ds_qa14:
        train_df['qa'] = 'qa14'
        test_df['qa'] = 'qa14'
    elif ds == ds_qa15:
        train_df['qa'] = 'qa15'
        test_df['qa'] = 'qa15'
    elif ds == ds_qa16:
        train_df['qa'] = 'qa16'
        test_df['qa'] = 'qa16'
    elif ds == ds_qa17:
        train_df['qa'] = 'qa17'
        test_df['qa'] = 'qa17'
    elif ds == ds_qa18:
        train_df['qa'] = 'qa18'
        test_df['qa'] = 'qa18'
    else:
        raise ValueError("Unknown dataset")    
    
    train_dfs.append(train_df)
    test_dfs.append(test_df)
    
    train_df = pd.DataFrame(ds['train'])
    test_df = pd.DataFrame(ds['test'])
    train_dfs.append(train_df)
    test_dfs.append(test_df)
    
# Combine all the dataframes into a single dataframe and shuffle the data
train_df = pd.concat(train_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

data_save_dir = 'data/babiqa'
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
    
train_df.to_csv(os.path.join(data_save_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(data_save_dir, 'test.csv'), index=False)