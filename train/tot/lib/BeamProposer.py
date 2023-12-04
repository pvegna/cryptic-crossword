from transformers import T5ForConditionalGeneration
from torch import no_grad, cat

class BeamProposer:

    def __init__(self, d, t):
        self.device = d
        self.tokenizer = t
        self.model = T5ForConditionalGeneration.from_pretrained(
            "/scratch/network/pvegna/models/flan-t5-large", 
            low_cpu_mem_usage=True).to(self.device)
        self.model.eval()
        self.C = 5
        self.prompt = self.tokenizer(['''question: What is the next step in 
                                      solving the cryptic crossword clue ''', 
                                      '''context: The last step was '''], 
                                      padding='None', truncation=False, 
                                      max_length=512, return_tensors='pt',
                                      return_attention_mask=False)['input_ids']

    def __call__(self, last, clue):
        input = self.tokenizer([f'"{last}"?', f'"{clue}".'], padding='None', 
                               truncation=False, max_length=512, return_tensors='pt', 
                               return_attention_mask=False)['input_ids']
        input = cat(self.prompt[0], input[0], self.prompt[1], input[1])
        with no_grad():
            gen = self.model.generate(input.to(self.device),
                             max_length=512, num_beams=self.C, 
                             num_return_sequences=self.C)
        return (gen, self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        