import json
import re

def extract(ex):
    ans = ""
    ans_id = re.findall(r"[A-Z\- ]+", ex)
    if len(ans_id) == 1:
        ans = ans_id[0]
    
    length = []
    for word in ans.split():
        length.append(len(word))
    return {'answer': ans, 'length': length}

filename = 'ab-50-test'

with open(filename+'.json', 'r', encoding='utf-8') as in_file:
    data = in_file.readlines()

with open(filename+'-processed.json', 'w', encoding='utf-8') as out_file:
    for line in data:
        input = json.loads(line)
        output = {}
        output['clue'] = input['input_ids'][0][input['input_ids'][0].index('context:')+8:]
        output['labels'] = extract(input['labels'][0])
        output['preds'] = extract(input['preds'][0])
        out_file.write(json.dumps(output)+'\n')