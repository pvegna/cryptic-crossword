import csv
import json
import re
from collections import OrderedDict

'''with open ('data/charades.csv', 'r') as charades_file:
    csv_file = csv.reader(charades_file)
    charade_freq = {}
    for charade in csv_file:
        text = (charade[1], charade[2])
        occurences = len(re.findall(r'\[[0-9]+\]', charade[3]))
        charade_freq[text] = occurences

charade_freq = OrderedDict(sorted(charade_freq.items(), key=lambda item: item[1], reverse=True))

with open ('data/charade_freq.csv', 'w') as freq_file:
    for (key, val) in charade_freq.items():
        c1, c2 = key
        freq_file.write(f'{c1}-->{c2},{val}\n')

with open ('data/indicators.csv', 'r') as indic_file:
    csv_file = csv.reader(indic_file)
    indic_freq = {}
    for indic in csv_file:
        text = (indic[1], indic[2])
        occurences = len(re.findall(r'\[[0-9]+\]', indic[3]))
        indic_freq[text] = occurences

indic_freq = OrderedDict(sorted(indic_freq.items(), key=lambda item: item[1], reverse=True))

with open ('data/indicator_freq.csv', 'w') as freq_file:
    for (key, val) in indic_freq.items():
        c1, c2 = key
        freq_file.write(f'{c1}-->{c2},{val}\n')

'''

'''

wp = ['alternation','anagram','container','deletion','hidden','homophone','insertion','reversal']
freq = [0] * 8

with open ('data/indicators_by_clue.csv', 'r') as indic_file:
    csv_indics = csv.reader(indic_file)
    for row in csv_indics:
        key = row[0]
        for i in range(1, len(row)):
            if row[i]:
                freq[i-1] += 1


import matplotlib.pyplot as plt


fig, ax = plt.subplots()
ax.pie(freq, labels=wp)
plt.show()

chars = []
freqs = []
with open('data/charade_freq.csv', 'r') as freq_file:
    csv_char = csv.reader(freq_file)
    for row in csv_char:
        chars.append(row[0])
        freqs.append(row[1])

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(values=['charade', 'freq'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[chars, freqs],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

fig.update_layout(width=750, height=1000)
fig.show()
'''