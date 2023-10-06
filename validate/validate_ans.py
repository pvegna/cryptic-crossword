from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import cuda, tensor, no_grad
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
data = load_dataset('json', data_files={'validate':'validate.json'}).shuffle()
data = data['validate'].select_columns(['clue', 'answer'])
data = data.map(preprocess, batched=True, batch_size=batch_size)
train_dataloader = DataLoader(data, #data.select_columns(['input_ids', 'attention_mask', 'labels']),
                               batch_size=batch_size, shuffle=False, 
                               collate_fn=collate)

epochs = 1

num_training_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

correct_preds = 0
total_targets = 0


model.eval()

with no_grad():
    for _ in range(epochs):
        for i, batch in enumerate(train_dataloader):
            print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.generate(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'] 
                        )
            
            # METRICS
            
            progress_bar.update()

model.save_pretrained('/scratch/network/pvegna/models/annotation_blind')