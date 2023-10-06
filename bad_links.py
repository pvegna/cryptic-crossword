import json
import re

with open ('data/consolidated.json', 'r') as in_file, open ('data/good_examples.json', 'w') as good_file, open ('data/bad_links.json', 'w') as bad_file:
    for line in in_file:
        bad_ex = False
        ex = json.loads(line)
        for charade in ex['charades']:
            #if not re.findall(r'\b' + re.escape(charade[0].lower()) + r'\b', ex['clue'].lower()):
            if not charade[0].lower() in ex['clue'].lower():
                bad_ex = True
                break
        for i in range(len(ex['indicators'])):
            resolved = False
            for split in ex['indicators'][i][0].lower().split('/'):
                #if re.findall(r'\b' + re.escape(split) + r'\b', ex['clue'].lower()):
                if split in ex['clue'].lower():
                    ex['indicators'][i] = [split, ex['indicators'][i][1]]
                    resolved = True      
            bad_ex = bad_ex or not resolved
            if bad_ex:
                break
        if bad_ex:
            bad_file.write(line)
        else:
            good_file.write(line)



