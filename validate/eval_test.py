from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-large-tokenizer")

model = T5ForConditionalGeneration.from_pretrained("/scratch/network/pvegna/models/tot-propose/checkpoint-1375", low_cpu_mem_usage=True).to(device)

'''clue = {"prompt":"question: What is the answer to the cryptic crossword clue? Use step-by-step reasoning. context: Prepare cooking here, putting in some corn (8)", 
        "answer": "REHEARSE", 
        "definition": "Prepare", 
        "charades": [["some corn", "EARS"]], 
        "indicators": [["cooking", "anagram"], ["putting in", "container"]]}'''

clue = {"prompt":'question: What is the next step in solving the cryptic crossword clue "Prepare cooking here, putting in some corn (8)"? context: The last step was "(cooking = anagram) here, putting in some corn = Prepare  (1 word of length 8)".'} 
        

input = tokenizer(clue['prompt'], padding='longest', truncation=True,
                      max_length=512, return_tensors='pt',
                      return_attention_mask=True)

gen_out = model.generate(input['input_ids'].to(device), max_length=512, num_beams=5, num_return_sequences=5)
print(gen_out)
print(tokenizer.batch_decode(gen_out))

