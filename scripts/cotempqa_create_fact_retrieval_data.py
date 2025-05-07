import pandas as pd
import ast

train_path = 'data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/train.csv'
test_path = 'data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# todo: add facts columns

# these are the columns in train data: index,question,reasoning,answer,messages. Answer is a list of strings where each string is present in a the list of sentences in the column facts. Identify those sentences and make a new column called 'answer_facts' and make a list of strings in that using these sentences.
train_data['facts'] = train_data['facts'].apply(ast.literal_eval)
train_data['answer'] = train_data['answer'].apply(ast.literal_eval)

train_data['answer_facts'] = train_data.apply(lambda row: [fact for fact in row['facts'] if any(answer in fact for answer in row['answer'])], axis=1)

test_data['facts'] = test_data['facts'].apply(ast.literal_eval)
test_data['answer'] = test_data['answer'].apply(ast.literal_eval)

test_data['answer_facts'] = test_data.apply(lambda row: [fact for fact in row['facts'] if any(answer in fact for answer in row['answer'])], axis=1)


# Save train_data to 'train_facts.csv'
train_data.to_csv('data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/train_facts.csv', index=False)

# Save test_data to 'test_facts.csv'
test_data.to_csv('data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/test_facts.csv', index=False)

print("train_facts.csv and test_facts.csv created successfully and saved in the same directory as the original files.")