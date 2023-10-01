from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import cuda
from tqdm.auto import tqdm


device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

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
train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

epochs = 1
optimizer = AdamW(model.parameters(), lr=3e-4)
num_training_steps = epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, 
    num_warmup_steps=num_training_steps*.10, 
    num_training_steps=num_training_steps)
progress_bar = tqdm(range(num_training_steps))

model.train()

for _ in range(epochs):
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels'], 
                    ).logits
        loss = F.cross_entropy(logits, batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update()

model.save_pretrained('models/annotation_blind')


