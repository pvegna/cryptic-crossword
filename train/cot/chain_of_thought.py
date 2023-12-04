from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import cuda, tensor
from tqdm.auto import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-small-tokenizer")
model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/flan-t5-small").to(device)

def preprocess(data):
    inputs = ["Solve the Cryptic Crossword clue: " + clue for clue in data['clue']]
    batch = tokenizer(inputs, 
                      padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)
    labels = []
    for i in range(len(data['answer'])):
        defin = "The definition is '" + data['definition'][i] + "'. "
        charades = ""
        for charade in data['charades'][i]:
            charades += charade[1] + " is a charade for '" + charade[0] + "'. "
        indics = ""
        for indic in data['indicators'][i]:
            indics += "'" + indic[0] + "' is an indicator of " + indic[1] + ". "
        labels.append(defin + charades + indics + "The answer is " + data['answer'][i] + ".")
    label_ids = tokenizer(labels,
                      padding='longest', truncation=True, 
                      return_tensors='pt',
                      return_attention_mask=True)
    batch['labels'] = label_ids['input_ids']
    for i in range(len(inputs)):
        print(inputs[i])
        print(batch['input_ids'][i])
        print(labels[i])
        print(batch['labels'][i])
    return batch

def collate(data):
    return {'input_ids': tensor([ex['input_ids'] for ex in data]), 
            'attention_mask': tensor([ex['attention_mask'] for ex in data]),
            'labels': tensor([ex['labels'] for ex in data])}

batch_size = 1#512
data = load_dataset('json', data_files={'train':'good_examples_edited.json'}).shuffle()
#data = data['train'].select_columns(['clue', 'answer'])
data = data['train'].select_columns(['clue', 'answer', 'definition', 'charades', 'indicators'])
data = data.map(preprocess, batched=True, batch_size=batch_size)
train_dataloader = DataLoader(data, #data.select_columns(['input_ids', 'attention_mask', 'labels']),
                               batch_size=batch_size, shuffle=False, 
                               collate_fn=collate)

epochs = 1
optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=.01)
num_training_steps = epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, 
    num_warmup_steps=num_training_steps*.10, 
    num_training_steps=num_training_steps)
loss_fn = CrossEntropyLoss()
progress_bar = tqdm(range(num_training_steps))

model.train()
with open ('/scratch/network/pvegna/cryptic/logs/cot.txt', 'w') as log_file:
    for _ in range(epochs):
        for i, batch in enumerate(train_dataloader):
            print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'], 
                        labels=batch['labels'], 
                        )
            
            logits = out.logits
            #print(logits.softmax(dim=1).shape)
            loss = loss_fn(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
            #loss = out.loss
            print("logits loss: " + str(loss.item()))
            print("default loss: " + str(out.loss.item()))
            #log_file.write(f'{loss.item()}\n')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update()

#model.save_pretrained('/scratch/network/pvegna/models/should_work-10000')


