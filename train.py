import transformers
import torch
import pandas as pd

def load_data(data_path):
    """
    Load the data
    """
    data = pd.read_csv(data_path)
    return data

def tokenize_data(data):
    """
    Tokenize the data
    """
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_data = []
    for sentence in data:
        tokenized_data.append(tokenizer.encode(sentence))
    return tokenized_data

