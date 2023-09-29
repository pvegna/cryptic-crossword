import csv
import json

'''
consol = {}
with open ('data/clues.csv', 'r', encoding='utf-8') as clues_file:
    csv_clues = csv.reader(clues_file)
    for row in csv_clues:
        consol[row[0]] = [row[1], row[2], row[3]]


with open ('data/charades_by_clue.csv', 'r') as charades_file:
    csv_chars = csv.reader(charades_file)
    for row in csv_chars:
        key = row[1]
        if key in consol.keys():
            consol[key].extend([row[2], row[3]])

for key, val in consol.items():
    if len(val) != 5:
        consol[key].extend([None, None])

wp = ['alternation','anagram','container','deletion','hidden','homophone','insertion','reversal']

with open ('data/indicators_by_clue.csv', 'r') as indic_file:
    csv_indics = csv.reader(indic_file)
    for row in csv_indics:
        key = row[0]
        if key in consol.keys():
            for i in range(1, len(row)):
                if row[i]:
                    consol[key].extend([row[i], wp[i-1]])

for key, val in consol.items():
    if len(val) != 7:
        consol[key].extend([None, None])

with open ('data/consolidated_pruned.json', 'w') as consol_file:
    for key, val in consol.items():
        line = {'rowid': key, 'clue': val[0], 'answer': val[1],
                'definition': val[2], 'charader': val[3], 'charade': val[4], 
                'wordplay': val[5], 'indicator': val[6]}
        if (line['charader'] or line['wordplay']):
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
'''
with open ('data/clues.csv', 'r', encoding='utf-8') as in_file, open ('data/clues_compressed.csv', 'w', encoding ='utf-8') as out_file:
    csv_in = csv.reader(in_file)
    for line in csv_in:
        out_file.write(','.join(line[:4]) + '\n')