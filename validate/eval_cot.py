from transformers import T5Tokenizer
from tqdm.auto import tqdm
import json
import numpy as np
import spacy

tokenizer = T5Tokenizer.from_pretrained("/scratch/network/pvegna/models/t5-large-tokenizer")
nlp = spacy.load('en_core_web_md')

with open('/scratch/network/pvegna/cryptic/cot-dicl-50-test-processed.json', 'r', encoding='utf-8') as in_file:
    text = in_file.readlines()
data = []
for line in text:
    data.append(json.loads(line))

num_training_steps = len(data)
progress_bar = tqdm(range(num_training_steps))

correct_tokens = {'answer': 0,
                  'definition': 0,
                  'charades': 0,
                  'indicators': 0}
max_acc_tokens = {'answer': 0,
                  'definition': 0,
                  'charades': 0,
                  'indicators': 0}
num_label_tokens = {'answer': 0,
                  'definition': 0,
                  'charades': 0,
                  'indicators': 0}
cosine_similarity = {'answer': 0,
                  'definition': 0,
                  'charades': 0,
                  'indicators': 0}
true_pos = {'charades': 0, 
            'indicators': 0}
false_pos = {'charades': 0, 
            'indicators': 0}
false_neg = {'charades': 0, 
            'indicators': 0}
length_percent_err = 0

def tokenize(ex):
    tokens = {}
    embeddings = {}
    for k,v in ex.items():
        if k != 'length':
            if k == 'charades' or k == 'indicators':
                toks = []
                embeds = []
                for item in v:
                    #toks.append(tokenizer(item, padding='longest', return_tensors='np')['input_ids'])
                    embeds.append([nlp(item[0]), nlp(item[1])])
                #tokens[k] = toks
                embeddings[k] = embeds
            else:
                tokens[k] = tokenizer(v, padding='longest', return_tensors='np')['input_ids']
                embeddings[k] = nlp(v)
    return tokens, embeddings

def mask_special_tokens(ex):
    pad_mask = np.where(ex != 0, True, False)
    eos_mask = np.where(ex != 1, True, False)
    return pad_mask & eos_mask

def accuracy(label, pred):
    label = label.flatten()
    pred = pred.flatten()
    pad_len = len(label)-len(pred)
    if pad_len > 0:
        pred= np.pad(pred, (0, pad_len), 'constant', constant_values=(0,0))
    elif pad_len < 0:
        label = np.pad(label, (0, -pad_len), 'constant', constant_values=(0,0))
    labels_mask = mask_special_tokens(label)
    preds_mask = mask_special_tokens(pred)
    correct = label[labels_mask] == pred[labels_mask]
    return (np.sum(correct), np.sum(labels_mask), np.sum(preds_mask))

def maximal_accuracy(label, pred):
    label = label.flatten()
    pred = pred.flatten()
    len_label = len(label)
    label = np.pad(label, (len(pred), 0), 'constant', constant_values=(0,0))
    pred = np.pad(pred, (0, len_label), 'constant', constant_values=(0,0))
    labels_mask = mask_special_tokens(label)
    max_correct = 0
    for i in range(len(label)):
        correct = label[labels_mask] == pred[labels_mask]
        max_correct = max(np.sum(correct), max_correct)
        pred = np.roll(pred, 1)
    return max_correct
  

for ex in data:
    labels, label_embeds = tokenize(ex['labels'])
    preds, pred_embeds = tokenize(ex['preds'])

    # answer accuracy 
    correct, num_labels, num_preds = accuracy(labels['answer'], preds['answer'])
    correct_tokens['answer'] += correct
    num_label_tokens['answer'] += num_labels
    
    # length percent error
    if num_labels:
        length_percent_err += abs(num_labels - num_preds) / num_labels

    # definition accuracy
    correct, num_labels, num_preds = accuracy(labels['definition'], preds['definition'])
    correct_tokens['definition'] += correct
    num_label_tokens['definition'] += num_labels

    # definition maximal accuracy
    max_acc_tokens['definition'] += maximal_accuracy(labels['definition'], preds['definition'])

    # answer cosine similarity
    cosine_similarity['answer'] += label_embeds['answer'].similarity(pred_embeds['answer'])

    # definition cosine similarity
    cosine_similarity['definition'] += label_embeds['definition'].similarity(pred_embeds['definition'])

    # indicator F1
    tp = 0
    for indic_l in ex['labels']['indicators']:
        for indic_p in ex['preds']['indicators']:
            if indic_l[0] == indic_p[0]:
                tp += 1
                break
    fn = len(ex['labels']['indicators']) - tp
    fp = len( ex['preds']['indicators']) - tp
    true_pos['indicators'] += tp
    false_neg['indicators'] += fn
    false_pos['indicators'] += fp

    # charade F1
    tp = 0
    for char_l in ex['labels']['charades']:
        for char_p in ex['preds']['charades']:
            if char_l[0] == char_p[0]:
                tp += 1
                break
    fn = len(ex['labels']['charades']) - tp
    fp = len( ex['preds']['charades']) - tp
    true_pos['charades'] += tp
    false_neg['charades'] += fn
    false_pos['charades'] += fp

    progress_bar.update()

print(f'accuracy: {correct_tokens["answer"] / num_label_tokens["answer"]}')
print(f'average length error: {length_percent_err / len(data)}')
print(f'cosine similarity: {cosine_similarity["answer"] / len(data)}')
print(f'definition accuracy: {correct_tokens["definition"] / num_label_tokens["definition"]}')
print(f'definition cosine similarity: {cosine_similarity["definition"] / len(data)}')
print(f'definition maximal rotated accuracy: {max_acc_tokens["definition"] / num_label_tokens["definition"]}')
print(f'indicator precision: {true_pos["indicators"] / (true_pos["indicators"] + false_pos["indicators"])}')
print(f'indicator recall: {true_pos["indicators"] / (true_pos["indicators"] + false_neg["indicators"])}')
print(f'charades precision: {true_pos["charades"] / (true_pos["charades"] + false_pos["charades"])}')
print(f'charades recall: {true_pos["charades"] / (true_pos["charades"] + false_neg["charades"])}')