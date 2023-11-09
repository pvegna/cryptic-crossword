from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_scheduler
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
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
                      padding="max_length", truncation=False,
                      max_length=130, return_tensors='pt',
                      return_attention_mask=True)
    #label = ["None" if not indic else indic[0][1] for indic in data['indicators']]
    labels = tokenizer(data['answer'], 
                      #label,
                      padding="max_length", truncation=False, 
                      max_length=50, return_tensors='pt',
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

def compute_metric():
    return

def collate(data):
    return {'input_ids': tensor([ex['input_ids'] for ex in data]), 
            'attention_mask': tensor([ex['attention_mask'] for ex in data]),
            'labels': tensor([ex['labels'] for ex in data])}

batch_size = 64
print('loading dataset...\n')
train_data = load_dataset('json', data_files={'train':'train.json'}).shuffle()
#data = data['train'].select_columns(['clue', 'answer'])
train_data = train_data['train'].select_columns(['clue', 'answer'])
train_data = train_data.map(preprocess, batched=True, batch_size=batch_size)

eval_data = load_dataset('json', data_files={'validate':'validate.json'}).shuffle()
eval_data = eval_data['validate'].select_columns(['clue', 'answer'])
eval_data = eval_data.map(preprocess, batched=True, batch_size=batch_size)
#train_dataloader = DataLoader(data, #data.select_columns(['input_ids', 'attention_mask', 'labels']),
#                               batch_size=batch_size, shuffle=False, 
#                               collate_fn=collate)
#collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", max_length=512)

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        return {'input_ids': tensor([ex['input_ids'] for ex in examples]), 
            'attention_mask': tensor([ex['attention_mask'] for ex in examples]),
            'labels': tensor([ex['labels'] for ex in examples])}

collator = DataCollator(tokenizer)

args = TrainingArguments(
    output_dir="/scratch/network/pvegna/cryptic/logs/annotation-blind-lg-t10/",
    per_device_train_batch_size=batch_size,
    learning_rate=5e-5,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_ratio=.10,
    logging_dir="/scratch/network/pvegna/cryptic/logs/annotation-blind-lg-t10/"
)

trainer = Trainer(model, 
                  args=args, 
                  train_dataset=train_data, 
                  eval_dataset=eval_data,
                  tokenizer=tokenizer, 
                  data_collator=collator)

trainer.train()

trainer.save_model('/scratch/network/pvegna/models/annotation-blind-lg-t10/')

