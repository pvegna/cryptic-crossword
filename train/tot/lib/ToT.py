from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch import cuda, no_grad

class Node:

    def __init__(self, pred, validity, parent = None):
        self.pred = pred
        self.parent = parent
        self.visited = 0
        self.children = []
        self.validity = validity
    
    def add_child(self, child):
        self.children.append(child)
        self.visited = 1
    
    
    
class ToT:

    def __init__(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        # might not need to send tokenizer to gpu
        self.tokenizer = T5Tokenizer.from_pretrained(
            "/scratch/network/pvegna/models/t5-large-tokenizer").to(self.device)
        self.proposer = BeamProposer(self.device, self.tokenizer)
        self.validator = RuleBasedValidator()
        
    def step(self, last, clue):
        proposal_ids, proposal_texts = self.proposer(last.pred, clue)
        for i in range(len(proposal_texts)):
            validity = self.validator(proposal_texts[i], clue)
            if validity >= 0:
                child = Node(proposal_ids[i], validity, parent=last)

                
        
        