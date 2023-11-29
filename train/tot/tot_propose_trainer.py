from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from torch import cuda
from tqdm.auto import tqdm
import numpy as np
import re

device = 'cuda' if cuda.is_available() else 'cpu'
print('loading tokenizer...\n')
tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-large-tokenizer")
print('loading model...\n')
model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/flan-t5-large", low_cpu_mem_usage=True)
print('load to device...\n')
model = model.to(device)

def preprocess(data):
    inputs = []
    for i in range(len(data["clue"])):
        prompt = f'question: What is the next step in solving the cryptic crossword clue "{data["clue"][i]}"? context: The last step was "{data["last"][i]}".'
        inputs.append(prompt)
    batch = {'input_ids': inputs, 'labels':data["label"]}
    return batch   

batch_size = 64
print('loading dataset...\n')
train_data = load_dataset('json', data_files={'train':'tot_propose_train.json'}).shuffle()
train_data = train_data['train'].select_columns(['clue', 'last', 'label'])
train_data = train_data.map(preprocess, batched=True, batch_size=batch_size)

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer([ex['input_ids'] for ex in examples], 
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
        labels = self.tokenizer([ex['labels'] for ex in examples],
                      padding='longest', truncation=True, 
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
        batch['labels'] = labels['input_ids']
        ignore_mask = labels == 0
        labels[ignore_mask] = -100
        batch['labels'] = labels['input_ids']
        return batch

collator = DataCollator(tokenizer)

args = TrainingArguments(
    output_dir="/scratch/network/pvegna/models/tot-propose/",
    per_device_train_batch_size=batch_size,
    learning_rate=5e-5,
    num_train_epochs=50,
    weight_decay=0.1,
    warmup_ratio=.10,
    logging_dir="/scratch/network/pvegna/cryptic/logs/tot-propose/",
    logging_steps=20,
    save_strategy="steps",
    save_steps=.5
)

trainer = Trainer(model, 
                  args=args, 
                  train_dataset=train_data,
                  tokenizer=tokenizer, 
                  data_collator=collator)

trainer.train()

