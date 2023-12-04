import json
import re

def extract(ex):
    definition = ""
    defin = re.findall(r"The definition is \"[^\"]*\"", ex)
    if len(defin) == 1:
        #defin = defin[0][defin[0].index("\"")+1:-1]
        defin = re.findall(r"\"[^\"]+\"$", defin[0])
        if len(defin) == 1:
            definition = defin[0][1:-1].strip()

    char_pairs = []
    charades = re.findall(r"[ A-Z]+ is a charade for \"[^\"]*\"", ex)
    for charade in charades:
        char_id = re.findall(r"\"[^\"]+\"$", charade)
        char = re.findall(r"^[ A-Z]+", charade)
        if len(char_id) == 1 and len(char) == 1:
            char_pairs.append([char_id[0][1:-1], char[0].strip()])

    indic_pairs = []    
    indicators = re.findall(r"\"[^\"]*\" is an indicator of [a-z]+", ex)
    for indicator in indicators:
        indic_id = re.findall(r"^\"[^\"]+\"", indicator)
        indic = re.findall(r"[a-z]+$", indicator)
        if len(indic_id) == 1 and len(indic) == 1:
            indic_pairs.append([indic_id[0][1:-1], indic[0]])

    ans = ""
    ans_id = re.findall(r"The answer is [A-Z\- ]+", ex)
    if len(ans_id) == 1:
        # does this find multi word answers?
        ans_id = re.findall(r"[A-Z\- ]+$", ans_id[0])
        if len(ans_id) == 1:
            ans = ans_id[0].strip()
    
    length = []
    for word in ans.split():
        length.append(len(word))
    return {'answer': ans, 'definition': definition, 'charades': char_pairs, 'indicators': indic_pairs, 'length': length}

filename = 'cot-dicl-50-test'

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