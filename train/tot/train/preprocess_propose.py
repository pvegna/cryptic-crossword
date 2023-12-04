import json

with open("gee_tot.json", "r", encoding="utf-8") as in_file:
    data = in_file.readlines()

inputs = []
for line in data:
    ex = json.loads(line)
    last = ex['clue']
    for step in ex['steps']:
        inputs.append({"clue": ex['clue'], 
                       "last": last, 
                       "label": step})
        last = step
    inputs.append({"clue": ex['clue'], 
                       "last": ex['steps'][-1], 
                       "label": ex['answer']})

with open("tot_propose_train.json", "w", encoding="utf-8") as out_file:
    for ex in inputs:
        out_file.write(json.dumps(ex) + '\n')
    