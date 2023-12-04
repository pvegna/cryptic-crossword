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
    inputs = ["question: What is the answer to the cryptic crossword clue? Use step-by-step reasoning. context: " + clue for clue in data['clue']]
    '''batch = tokenizer(inputs, 
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)'''
    labels = []
    for i in range(len(data['answer'])):
        length = ''
        l = re.findall(r'\([0-9,\-\s]*\)', data['clue'][i])
        if l:
            l = re.sub(r'[\(\)]', '', l[0], 2)
            word_count = len(re.findall(r'[,\-]', l)) + 1
            if word_count == 1:
                length = f'The answer is 1 word of length {l}. '
            else:
                length = f'The answer is {word_count} words of lengths {l}. '
        defin = ''
        if data['definition'][i]:
            defin = 'The definition is "' + data['definition'][i] + '". '
        charades = ''
        if data['charades'][i]:
            for charade in data['charades'][i]:
                charades += charade[1] + ' is a charade for "' + charade[0] + '". '
        indics = ''
        if data['indicators'][i]:
            for indic in data['indicators'][i]:
                if indic:
                    indics += '"' + indic[0] + '" is an indicator of ' + indic[1] + '. '
        # original order:
        #labels.append(length + defin + charades + indics + 'The answer is ' + data['answer'][i] + '.')
        labels.append(defin + indics + charades + length + 'The answer is ' + data['answer'][i] + '.')
    batch = {'input_ids': inputs, 'labels':labels}
    return batch

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    

batch_size = 64
print('loading dataset...\n')
train_data = load_dataset('json', data_files={'train':'good_examples_edited.json'}).shuffle()
train_data = train_data['train'].select_columns(['clue', 'answer', 'definition', 'charades', 'indicators'])
train_data = train_data.map(preprocess, batched=True, batch_size=batch_size)

eval_data = load_dataset('json', data_files={'validate':'validate.json'}).shuffle()
eval_data = eval_data['validate'].select_columns(['clue', 'answer', 'definition', 'charades', 'indicators'])
eval_data = eval_data.map(preprocess, batched=True, batch_size=batch_size)


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
    output_dir="/scratch/network/pvegna/models/cot-dicl/",
    per_device_train_batch_size=batch_size,
    learning_rate=5e-5,
    num_train_epochs=50,
    weight_decay=0.1,
    warmup_ratio=.10,
    logging_dir="/scratch/network/pvegna/cryptic/logs/cot-dicl/",
    logging_steps=20,
    save_strategy="steps",
    save_steps=.5
)

trainer = Trainer(model, 
                  args=args, 
                  train_dataset=train_data, 
                  eval_dataset=eval_data,
                  tokenizer=tokenizer, 
                  data_collator=collator)

trainer.train()

#trainer.save_model('/scratch/network/pvegna/models/cot-len-100/')

