import torch
from torch import nn
import json
from itertools import chain
import struct
from functools import reduce
import numpy as np


def convert_model(path):
    model = torch.load(f'{path}/pytorch_model.bin')
    with open('model.bin', 'wb') as f:
        f.write(b"start")
        f.write(struct.pack('I', len(model)))
        for i, (k, v) in enumerate(model.items()):
            s = v.shape
            print(i, k, s)
            f.write(struct.pack("I", len(s)))
            # writing dimensions
            for d in v.shape: f.write(struct.pack("I", d))
            v.cpu().numpy().tofile(f)
    return v

def test_model():
    with open('model.bin', 'rb') as f:
        o = f.read(5)
        n_layers = struct.unpack("I", f.read(4))[0]
        for l in range(n_layers):
            layer_ndim = struct.unpack("I", f.read(4))[0]
            layer_dim = struct.unpack("I"*layer_ndim, f.read(4*layer_ndim)) 
            res = reduce(lambda x, y: x*y, layer_dim)
            layer = np.frombuffer(f.read(res * 4), dtype=np.float32)
    return layer


def convert_tokenizer(path):
    with open(f'{path}/vocab.json', encoding='utf-8') as f: o = json.load(f)
    with open(f'{path}/tokenizer.json', encoding='utf-8') as f: tokenizer_json = json.load(f)

    tokens, scores = [], []
    for j,i in o.items():
        j = j.replace('_', ' ')
        j = j.encode('utf-8')
        tokens.append(j)
        scores.append(0.0)

    max_token_length = max(map(len, tokens))

    with open('tokenizer.bin', 'wb') as f:
        f.write(struct.pack("I", max_token_length))
        for bytes, scores in zip(tokens, scores):
            f.write(struct.pack("fI", scores, len(bytes)))
            f.write(bytes)

    print("tokenizer exported")

convert_tokenizer("model")
d1 = convert_model('model')
d2 = test_model()
print(d1.numpy(), d1.dtype)
print(d2, d2.dtype)
