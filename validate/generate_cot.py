from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import cuda, tensor, no_grad
from tqdm.auto import tqdm
import json
import re


device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-large-tokenizer")
#model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/good-examples-1000/").to(device)

model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/cot-dicl/checkpoint-500/", low_cpu_mem_usage=True).to(device)

def preprocess(data):
    inputs = ["question: What is the answer to the cryptic crossword clue? Use step-by-step reasoning. context: " + clue for clue in data['clue']]
    batch = tokenizer(inputs, 
                      #["Extractive QA: Context: " + clue + "Question: What is the person's age?" for clue in data['clue']],
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=False)
    labels = []
    # add step for length?
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
        labels.append(length + defin + charades + indics + 'The answer is ' + data['answer'][i] + '.')
    labels = tokenizer(labels, 
                      #label,
                      padding='longest', truncation=True, 
                      return_tensors='pt',
                      return_attention_mask=False)
    ignore_mask = labels == 0
    labels[ignore_mask] = -100
    batch['labels'] = labels['input_ids']
    return batch

def collate(data):
    return {'input_ids': tensor([ex['input_ids'] for ex in data]), 
            'labels': tensor([ex['labels'] for ex in data])}

batch_size = 1
data = load_dataset('json', data_files={'test':'test_edited.json'}).shuffle()
data = data['test'].select_columns(['clue', 'answer', 'definition', 'charades', 'indicators'])
data = data.map(preprocess, batched=True, batch_size=batch_size)
train_dataloader = DataLoader(data, #data.select_columns(['input_ids', 'attention_mask', 'labels']),
                               batch_size=1, shuffle=False, 
                               collate_fn=collate)

epochs = 1

num_training_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

correct_preds = 0
total_targets = 0


model.eval()

with open ('/scratch/network/pvegna/cryptic/logs/cot-dicl-50-test.json', 'w') as out_file: #, open ('/scratch/network/pvegna/cryptic/logs/annotation-blind-10-eval.txt', 'w') as eval_file:
    with no_grad():
        for _ in range(epochs):
            for i, batch in enumerate(train_dataloader):
                #print(batch)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model.generate(input_ids=batch['input_ids'], max_length=512)
                decode = {'input_ids': batch['input_ids'], 'labels': batch['labels'], 'preds': out}
                batch = {k: tokenizer.batch_decode(v) for k, v in decode.items()}
                out_file.write(json.dumps(batch)+'\n')
                #print(batch)
                # METRICS
                progress_bar.update()