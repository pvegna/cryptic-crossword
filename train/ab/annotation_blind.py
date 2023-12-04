from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import cuda, tensor
from tqdm.auto import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'
print('loading tokenizer...\n')
tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-large-tokenizer")
print('loading model...\n')
model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/flan-t5-large", low_cpu_mem_usage=True)
print('load to device...\n')
model = model.to(device)

def preprocess(data):
    inputs = ["question: What is the answer to the cryptic crossword clue? context: " + clue for clue in data['clue']]
    batch = tokenizer(inputs, 
                      #["Extractive QA: Context: " + clue + "Question: What is the person's age?" for clue in data['clue']],
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
    #label = ["None" if not indic else indic[0][1] for indic in data['indicators']]
    labels = tokenizer(data['answer'], 
                      #label,
                      padding='longest', truncation=True, 
                      return_tensors='pt',
                      return_attention_mask=True)
    ignore_mask = labels == 0
    labels[ignore_mask] = -100
    batch['labels'] = labels['input_ids']
    #for i in range(len(inputs)):
    #    print(inputs[i])
    #    print(batch['input_ids'][i])
    #    print(data['answer'][i])
    #    print(batch['labels'][i])
    return batch

def collate(data):
    return {'input_ids': tensor([ex['input_ids'] for ex in data]), 
            'attention_mask': tensor([ex['attention_mask'] for ex in data]),
            'labels': tensor([ex['labels'] for ex in data])}

batch_size = 64
print('loading dataset...\n')
data = load_dataset('json', data_files={'train':'train.json'}).shuffle()
#data = data['train'].select_columns(['clue', 'answer'])
data = data['train'].select_columns(['clue', 'answer'])
data = data.map(preprocess, batched=True, batch_size=batch_size)
train_dataloader = DataLoader(data, #data.select_columns(['input_ids', 'attention_mask', 'labels']),
                               batch_size=batch_size, shuffle=False, 
                               collate_fn=collate)

epochs = 10
optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=.01)
num_training_steps = epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, 
    num_warmup_steps=num_training_steps*.10, 
    num_training_steps=num_training_steps)
loss_fn = CrossEntropyLoss()
progress_bar = tqdm(range(num_training_steps))

model.train()
with open ('/scratch/network/pvegna/cryptic/logs/annotation-blind-lg-10.txt', 'w') as log_file:
    for _ in range(epochs):
        for i, batch in enumerate(train_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch['labels']
                        )
            
            logits = out.logits
            #print(logits.softmax(dim=1).shape)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
            #loss = out[0]
            log_file.write(f'{loss.item()}\n')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update()

model.save_pretrained('/scratch/network/pvegna/models/annotation-blind-lg-10')


