from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import cuda
from torch import tensor
from tqdm.auto import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-small-tokenizer")
model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/flan-t5-small").to(device)

def preprocess(data):
    batch = tokenizer(["Cryptic Crossword: " + clue for clue in data['clue']], 
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
    labels = tokenizer(data['answer'], 
                      padding='longest', truncation=True, 
                      return_tensors='pt',
                      return_attention_mask=True)
    batch['labels'] = labels['input_ids']
    return batch

def collate(data):
    return {'input_ids': tensor([ex['input_ids'] for ex in data]), 
            'attention_mask': tensor([ex['attention_mask'] for ex in data]),
            'labels': tensor([ex['labels'] for ex in data])}

batch_size = 64
data = load_dataset('json', data_files={'train':'train.json'}).shuffle()
data = data['train'].select_columns(['clue', 'answer'])
data = data.map(preprocess, batched=True, batch_size=batch_size)
train_dataloader = DataLoader(data, #data.select_columns(['input_ids', 'attention_mask', 'labels']),
                               batch_size=batch_size, shuffle=False, 
                               collate_fn=collate)

epochs = 1
optimizer = AdamW(model.parameters(), lr=3e-4)
num_training_steps = epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, 
    num_warmup_steps=num_training_steps*.10, 
    num_training_steps=num_training_steps)
loss_fn = CrossEntropyLoss()
progress_bar = tqdm(range(num_training_steps))

model.train()
with open ('/scratch/network/pvegna/cryptic/logs/annotation-blind-loss.txt', 'w') as log_file:
    for _ in range(epochs):
        for i, batch in enumerate(train_dataloader):
            print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch['labels'], 
                        ).logits
            #print(logits.view(-1, logits.size(-1)).shape)
            #print(batch['labels'].view(-1).shape)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
            log_file.write(f'{loss.item()}\n')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update()

model.save_pretrained('/scratch/network/pvegna/models/annotation_blind')


