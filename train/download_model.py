from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoConfig

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# tokenizer.save_pretrained('/scratch/network/pvegna/models/t5-small-tokenizer')

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
model.save_pretrained('/scratch/network/pvegna/models/flan-t5-small')