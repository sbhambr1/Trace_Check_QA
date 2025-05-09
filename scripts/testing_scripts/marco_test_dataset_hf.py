import os
import sys
import warnings
import pandas as pd
from datasets import Dataset, load_dataset

warnings.filterwarnings("ignore")

ds = load_dataset("microsoft/ms_marco", "v2.1")

print("Loading dataset...")

# DatasetDict({
#     validation: Dataset({
#         features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],
#         num_rows: 101093
#     })
#     train: Dataset({
#         features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],
#         num_rows: 808731
#     })
#     test: Dataset({
#         features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],
#         num_rows: 101092
#     })
# })

# ['DESCRIPTION', 'PERSON', 'ENTITY', 'LOCATION', 'NUMERIC']



# Get 1000 samples of each query_type from the train dataset
query_types = ["DESCRIPTION", "ENTITY", "NUMERIC", "LOCATION", "PERSON"]

test_samples = []
train_samples = []
validation_samples = []

for query_type in query_types:
    test_samples.append(ds["test"].filter(lambda example: example["query_type"] == query_type).shuffle().select(range(200)).to_pandas())
    train_samples.append(ds["train"].filter(lambda example: example["query_type"] == query_type).shuffle().select(range(1000)).to_pandas())
    validation_samples.append(ds["validation"].filter(lambda example: example["query_type"] == query_type).shuffle().select(range(200)).to_pandas())
    
test_samples = Dataset.from_pandas(pd.concat(test_samples))
train_samples = Dataset.from_pandas(pd.concat(train_samples))
validation_samples = Dataset.from_pandas(pd.concat(validation_samples))

data_save_dir = "data/marcoqa"
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)

# Save the samples as CSV files
train_samples.to_csv("data/marcoqa/train.csv", index=False)
test_samples.to_csv("data/marcoqa/test.csv", index=False)
validation_samples.to_csv("data/marcoqa/validation.csv", index=False)
