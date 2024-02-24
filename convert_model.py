import torch
from torch import nn
import json
from itertools import chain
import struct

model = torch.load('model/pytorch_model.bin')

for k, v in model.items():
    # print(k, v.shape)
    pass

def convert_tokenizer():
    with open('model/vocab.json', encoding='utf-8') as f: o = json.load(f)
    with open('model/tokenizer.json', encoding='utf-8') as f: tokenizer_json = json.load(f)
    print(tokenizer_json['model']['vocab'].__len__())
    return o

vocab = convert_tokenizer()

tokens, scores = [], []
for j,i in vocab.items():
    j = j.replace('_', ' ')
    j = j.encode('utf-8')
    tokens.append(j)
    scores.append(0.0)

max_token_length = max(map(len, tokens))
print(len(tokens))

with open('tokenizer.bin', 'wb') as f:
    f.write(struct.pack("I", max_token_length))
    for bytes, scores in zip(tokens, scores):
        f.write(struct.pack("fI", scores, len(bytes)))
        f.write(bytes)

print("tokenizer exported")
