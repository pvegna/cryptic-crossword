from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoConfig

#tokenizer = T5Tokenizer.from_pretrained("t5-large")
#tokenizer.save_pretrained('/scratch/network/pvegna/models/t5-large-tokenizer')

model = T5ForConditionalGeneration.from_pretrained("t5-large")
model.save_pretrained('/scratch/network/pvegna/models/t5-large')