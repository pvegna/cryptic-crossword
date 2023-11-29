import json
import re
import random

with open('gee_tot.json', 'r', encoding="utf-8") as in_file:
    data = in_file.readlines()


steps = []
for line in data:
    ex = json.loads(line)
    clue = ex["clue"]
    step = ex["clue"]
    
    # expand length
    length = ''
    l = re.findall(r'\([0-9,\-\s]*\)', step)
    if l:
        step = step[:step.index(l[0])].strip()
        l = re.sub(r'[\(\)]', '', l[0], 2)
        word_count = len(re.findall(r'[,\-]', l)) + 1
        if word_count == 1:
            length = f'  (1 word of length {l})'
        else:
            length = f' ({word_count} words of lengths {l})'
    steps.append({"clue": clue, "context": "1. " + step + length, "labels": "1"})

    # -------------------definition noise-----------------------------------------------#
    defin = ex["definition"]
    front = True if re.findall(r'^' + defin, step) else False
    def repl_trim(s, r):
        s = s.replace(r, "")
        # trim excess punctuation/whitespace
        s = re.sub(r'^[\s\.\?!,;:]+', '', s)
        s = re.sub(r'[\s\.\?!,;:]+$', '', s)
        return s
    step = repl_trim(step, defin)

    num_words = step.count(" ")
    ok_i = min(random.randint(1, 3), num_words-1)
    bad_i = min(random.randint(1, 3), num_words-1)
    split_step = step.split(" ")
    if front:
        ok_def = " ".join(split_step[ok_i:])
        bad_def = " ".join(split_step[:bad_i])
        ok_step = repl_trim(step, ok_def)
        bad_step = repl_trim(step, bad_def)
        ok_step = defin + " " + ok_step + " = " + ok_def + length
        bad_step = defin + " " + bad_step + " = " + bad_def + length
    else:
        ok_def = " ".join(split_step[:ok_i])
        bad_def = " ".join(split_step[bad_i:])
        ok_step = repl_trim(step, ok_def)
        bad_step = repl_trim(step, bad_def)
        ok_step = ok_step + " " + defin + " = " + ok_def + length
        bad_step = bad_step + " " + defin + " = " + bad_def + length
    
    good_step = step + " = " + defin + length
    order = [1, 2, 3]
    random.shuffle(order)
    def_cont = ["1. ", "2. ", "3. "]
    def_cont[order[0]-1] += good_step
    def_cont[order[1]-1] += ok_step
    def_cont[order[2]-1] += bad_step

    steps.append({"clue": clue, "context": "\n".join(def_cont), "labels": ",".join(str(elem) for elem in order)})
    step = good_step
    
    # ------------------------indicator noise--------------------------------------------#
    indic_options = ["anagram", "container", "combination", "deletion", "reversal", "homophone", "acronym"]
    for indic in ex["indicators"]:
        if indic:
            r = random.uniform(0,1)
            if indic[0] in step:
                to_replace = indic[0]
            else:
                to_replace = indic[0][0].upper() + indic[0][1:] 
            good_step = re.sub(r'\b' + to_replace + r'\b', f"({indic[0]} = {indic[1]})", step)
            ok_step = re.sub(r'\b' + to_replace + r'\b', f"({indic[0]} = {indic_options[random.randint(0, 6)]})", step)
            if r < .33:
                bad_step = re.sub(r'\b' + to_replace + r'\b', indic[0].upper(), step)
            

            

                
            #steps.append(step)

    '''
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
    
    '''  
with open('tot_vote_train.json', 'w') as out_file:
    for ex in steps:
        out_file.write(json.dumps(ex) + '\n')




            
        
        
