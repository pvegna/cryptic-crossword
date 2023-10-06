import csv
import json
import random 
import math
'''

consol = {}
with open ('data/clues_large.csv', 'r', encoding='utf-8') as clues_file:
    csv_clues = csv.reader(clues_file)
    for row in csv_clues:
        consol[row[0]] = [row[1], row[2], row[3], [], []]


with open ('data/charades_by_clue.csv', 'r') as charades_file:
    csv_chars = csv.reader(charades_file)
    for row in csv_chars:
        key = row[1]
        if key in consol.keys():
            consol[key][3].append((row[2], row[3]))


wp = ['alternation','anagram','container','deletion','hidden','homophone','insertion','reversal']

with open ('data/indicators_by_clue.csv', 'r') as indic_file:
    csv_indics = csv.reader(indic_file)
    for row in csv_indics:
        key = row[0]
        if key in consol.keys():
            for i in range(1, len(row)):
                if row[i]:
                    consol[key][4].append((row[i], wp[i-1]))

with open ('data/consolidated_pruned.json', 'w') as consol_file:
    for key, val in consol.items():
        line = {'rowid': key, 'clue': val[0], 'answer': val[1],
                'definition': val[2], 'charades': val[3], 'indicators': val[4]}
        if (line['charades'] or line['indicators']):
            consol_file.write(json.dumps(line) + '\n')


with open ('data/consolidated.json', 'r') as data_file:
    consol_data = data_file.readlines()


rowids, clues, ans, defs, chars, char_ans, wps, indics = [], [], [], [], [], [], [], []
for line in consol_data:
    fields = json.loads(line)
    rowids.append(fields['rowid'])
    clues.append(fields['clue'])
    ans.append(fields['answer'])
    defs.append(fields['definition'])
    chars.append(fields['charader'])
    char_ans.append(fields['charade'])
    wps.append(fields['wordplay'])
    indics.append(fields['indicator'])


import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(values=['rowid', 'clue', 'answer',
                'definition', 'charader', 'charade', 
                'wordplay', 'indicator'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[rowids, clues, ans, defs, chars, char_ans, wps, indics],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=1500, height=1000)
fig.show()


with open ('data/clues.csv', 'r', encoding='utf-8') as in_file, open ('data/clues_compressed.csv', 'w', encoding ='utf-8') as out_file:
    csv_in = csv.reader(in_file)
    for line in csv_in:
        out_file.write(','.join(line[:4]) + '\n')'''

'''
with open ('data/good_examples_raw.json', 'r') as in_file:
    in_data = in_file.readlines()
    
random.shuffle(in_data)
train_len = math.ceil((len(in_data)) * .80) - 200
test_len = train_len + math.ceil((len(in_data)) * .15)

with open ('data/train.json', 'w') as train_file:
    train_file.writelines(in_data[:train_len])
with open ('data/test.json', 'w') as test_file:
    test_file.writelines(in_data[train_len:test_len])
with open ('data/validate.json', 'w') as valid_file:
    valid_file.writelines(in_data[test_len:])

def remove_insert(split):
    with open (f'data/{split}.json', 'r') as in_file, open (f'data/{split}1.json', 'w') as out_file:
        for line in in_file:
            line = json.loads(line)
            for i in range(len(line['indicators'])):
                if line['indicators'][i][1] == 'insertion':
                    line['indicators'][i][1] = 'container'
            out_file.write(json.dumps(line)+'\n')

remove_insert('train')
remove_insert('validate')
remove_insert('test')
'''
with open ('data/train1.json', 'r') as in_file:
    in_data = in_file.readlines()
    
random.shuffle(in_data)

with open ('data/train2.json', 'w') as train_file:
    train_file.writelines(in_data)
