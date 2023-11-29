import json
import re

with open('good_examples_edited.json', 'r', encoding="utf-8") as in_file:
    data = in_file.readlines()

with open('gee_tot.json', 'w') as out_file:
    for line in data:
        ex = json.loads(line)
        steps = []
        
        step = ex["clue"]
        
        # expand length
        length = ''
        l = re.findall(r'\([0-9,\-\s]*\)', step)
        if l:
            step = step[:step.index(l[0])]
            l = re.sub(r'[\(\)]', '', l[0], 2)
            word_count = len(re.findall(r'[,\-]', l)) + 1
            if word_count == 1:
                length = f'  (1 word of length {l})'
            else:
                length = f' ({word_count} words of lengths {l})'
        steps.append(step + length)

        # move definition to end
        defin = ex["definition"]
        step = step.replace(defin, "")
        # trim excess punctuation/whitespace
        step = re.sub(r'^[\s\.\?!,;:]+', '', step)
        step = re.sub(r'[\s\.\?!,;:]+$', '', step)
        step += " = " + defin + length
        steps.append(step)

        # replace indicators
        for indic in ex["indicators"]:
            if indic:
                if indic[0] in step:
                    step = re.sub(r'\b' + indic[0] + r'\b', f"({indic[0]} = {indic[1]})", step)
                else:
                    step = re.sub(r'\b' + indic[0][0].upper() + indic[0][1:] + r'\b', f"({indic[0]} = {indic[1]})", step)
                #step = step.replace(indic[0], f"({indic[0]} = {indic[1]})")
                steps.append(step)

        # replace charades
        for char in ex["charades"]:
            if char:
                if char[0] in step:
                    step = re.sub(r'\b' + char[0] + r'\b', char[1], step)
                else:
                    step = re.sub(r'\b' + char[0][0].upper() + char[0][1:] + r'\b', char[1], step)
                #step = step.replace(char[0], char[1])
                steps.append(step)
        
        # answer step
        steps.append(ex["answer"] + " = " + defin + length)

        data = ex
        data["steps"] = steps
        out_file.write(json.dumps(data) + '\n')        

