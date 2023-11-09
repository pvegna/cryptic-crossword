import re

ex = {"input_ids": "question: What is the answer to the cryptic crossword clue? Use step-by-step reasoning. context: Prepare to fight again behind leader of militia (5)</s>", "labels": "The definition is 'Prepare to fight again'. REAR is a charade for 'behind'. The answer is REARM.</s>", "preds": "<pad> The definition is 'Prepare to fight again'. RE is a charade for 'behind'. 'leader of' is an indicator of acronym. 'behind' is an indicator of combination. The answer is REARM.</s>"}

clue = ex['input_ids'][ex["input_ids"].index('context:')+8:]
defin = re.findall(r"The definition is '[^']*'", ex['preds'])
charades = re.findall(r"[A-Z]* is a charade for '[^']*'", ex['preds'])
indics = re.findall(r"'[^']*' is an indicator of [a-z]*", ex['preds'])
ans = re.findall(r"The answer is [A-Z]*", ex['preds'])

print("clue: ")
print(clue)
print("defs: ")
print(defin)
print("charades: ")
print(charades)
print("indicators: ")
print(indics)
print("answer: ")
print(ans)

def_id = defin[0][defin[0].index("'")+1:-1]
char_id = re.findall(r"'[^']+'$", charades[0])
char = re.findall(r"^[A-Z]+", charades[0])
indic_id = re.findall(r"^'[^']+'", indics[0])
indic = re.findall(r"[a-z]+$", indics[0])
ans_id = re.findall(r"[A-Z]+$", ans[0])

print("def id: ")
print(def_id)
print("charades: ")
print([char_id, char])
print("indicators: ")
print([indic_id, indic])
print("answer: ")
print(ans_id)