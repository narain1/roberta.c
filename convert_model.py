import torch
from torch import nn
import json
from itertools import chain
import struct
from functools import reduce
import numpy as np

config = {
    'vocab_size': 30522,
    'hidden_size': 768,
    'intermediate_size': 3072,
    'n_max_tokens' : 512,
    'n_attention_heads': 12,
    'n_hidden_layers': 12,
}

config_vars = list(config.keys())

class Tensor:
    def __init__(self, f, *shape):
        assert all(isinstance(x, int) for x in shape)
        self.shape = shape
        self.load_weights(f)

    def __repr__(self):
        return f'{self.shape}'

    def load_weights(self, file):
        res = reduce(lambda x, y: x*y, self.shape)
        self.data = np.frombuffer(file.read(res * 4), dtype=np.float32)

class EmbeddingLayer:
    def __init__(self, file, _config):
        self.word_emb = Tensor(file, _config['vocab_size'], _config['hidden_size'])
        self.pos_emb = Tensor(file, _config['n_max_tokens'], _config['hidden_size'])
        self.tok_type_w = Tensor(file, 2, _config['hidden_size'])
        self.ln = LayerNorm(file, _config['hidden_size'])

class Linear:
    def __init__(self, file, f_in, f_out, bias=True):
        self.w = Tensor(file, f_in, f_out)
        self.b = Tensor(file, f_in)

class LayerNorm:
    def __init__(self, file, hidden_size):
        self.gamma = Tensor(file, hidden_size)
        self.beta = Tensor(file, hidden_size)

class EncoderLayer:
    def __init__(self, file, hidden_size, ff_dim):
        self.query = Linear(file, hidden_size, hidden_size)
        self.key = Linear(file, hidden_size, hidden_size, bias=True)
        self.value = Linear(file, hidden_size, hidden_size)
        self.ff_in = Linear(file, ff_dim, hidden_size)
        self.ff_out = Linear(file, hidden_size, ff_dim)
        self.ln = LayerNorm(file, hidden_size)

class ClsLayer:
    def __init__(self, file, hidden_size, vocab_size):
        self.pred_bias = Tensor(file, vocab_size)
        self.transform = Linear(file, hidden_size, hidden_size)
        self.ln = LayerNorm(file, hidden_size)
        self.decoder = Tensor(file, vocab_size, hidden_size)
        self.seq = Linear(file, 2, hidden_size)


class RobertaModel:
    def __init__(self, file, _config):
        self.emb = EmbeddingLayer(file, _config)
        self.EncoderLayers = [EncoderLayer(file,
                                           _config['hidden_size'],
                                           _config['intermediate_size'])
                              for _ in range(_config['n_hidden_layers'])
                             ]
        self.pool = Linear(file, _config['hidden_size'], _config['hidden_size'])
        self.cls = ClsLayer(file, 2, _config['hidden_size'])

def convert_model(path):
    model = torch.load(f'{path}/pytorch_model.bin')
    rev_config = {j:i for i,j in config.items()}
    print(model['bert.embeddings.word_embeddings.weight'].flatten()[:5])
    print(model['bert.embeddings.position_embeddings.weight'].flatten()[:5])
    print(model['bert.encoder.layer.0.attention.self.query.weight'].flatten()[:5])
    print(model['bert.embeddings.LayerNorm.gamma'].flatten()[:5])
    print(model['bert.embeddings.LayerNorm.beta'].flatten()[:5])
    print(model['bert.encoder.layer.0.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.1.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.2.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.3.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.4.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.5.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.6.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.7.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.8.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.9.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.10.attention.self.query.weight'].flatten()[:5])
    print(model['bert.encoder.layer.11.attention.self.query.weight'].flatten()[:5])
    print(model['bert.pooler.dense.weight'].flatten()[:5])
    print(model['cls.predictions.decoder.weight'].flatten()[:5])

    with open('model.bin', 'wb') as f:
        f.write(struct.pack('I', len(config)))
        for ck, cv in config.items():
            f.write(struct.pack('I', cv))
        for i, (k, v) in enumerate(model.items()):
            # if k.startswith('cls'): continue
            # print(i, k, v.shape)
            # assert not any(ss is None for ss in s), f"failed in config creation {v.shape}, {config}"
            if "embeddings" not in k and len(v.shape) == 2:
                v.cpu().T.numpy().tofile(f)
            else:
                v.cpu().numpy().tofile(f)

def test_model():
    with open('model.bin', 'rb') as f:
        conf_size = struct.unpack("I", f.read(4))[0]
        _config = {ck: struct.unpack("I", f.read(4))[0] for ck in config_vars}

        model = RobertaModel(f, _config)

    return model


def convert_tokenizer(path):
    #with open(f'{path}/vocab.json', encoding='utf-8') as f: o = json.load(f)
    with open(f'{path}/tokenizer.json', encoding='utf-8') as f: tokenizer_json = json.load(f)

    tokens, scores = [], []
    for j,i in tokenizer_json['model']['vocab'].items():
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


convert_tokenizer("model")
print('tokenizer exported')

convert_model('model')
d2 = test_model()
print('model exported')
