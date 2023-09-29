from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def preprocess(data):
    batch = tokenizer("Cryptic Crossword: " + data['clue'], 
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
    labels = tokenizer(data['answer'], 
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
    batch['labels'] = labels['input_ids']
    return batch

batch_size = 64
data = load_dataset('json', data_files='train.json')
data = data.map(preprocess, batched=True, batch_size=batch_size)

epochs = 1
lr = 3e-5


