from __future__ import print_function
from glob import glob
from torch.utils.serialization import load_lua
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import _pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import warnings
import time

warnings.filterwarnings("ignore")

torch_train = glob('data/train/*.torch')
torch_test = glob('data/test/*.torch')

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class PathRNN(nn.Module):
    def __init__(self,
                 entity_vocab_size: int,
                 entity_type_size: int,
                 relation_vocab_size: int,
                 entity_vocab_dim: int,
                 entity_type_dim: int,
                 relation_vocab_dim: int,
                 rnn_type: str,
                 mlp_hidden: int,
                 rnn_hidden_dim: int):
        super(PathRNN, self).__init__()
        self.ev_dim = entity_vocab_dim
        self.et_dim = entity_type_dim
        self.rv_dim = relation_vocab_dim
        self.rnn_type = rnn_type

        self.rnn_input = self.ev_dim + self.et_dim + self.rv_dim

        self.ev_embedder = nn.Embedding(entity_vocab_size, entity_vocab_dim)
        self.et_embedder = nn.Embedding(entity_type_size, entity_type_dim)
        self.rf_embedder = nn.Embedding(relation_vocab_size, relation_vocab_dim)
        self.rb_embedder = nn.Embedding(relation_vocab_size, relation_vocab_dim)

        self.f_rnn = getattr(nn, rnn_type)(self.rnn_input,
                                           rnn_hidden_dim,
                                           num_layers = 1,
                                           batch_first=True)

        self.b_rnn = getattr(nn, rnn_type)(self.rnn_input,
                                           rnn_hidden_dim,
                                           num_layers = 1,
                                           batch_first=True)

        self.proj = nn.Sequential(nn.Linear(2 * rnn_hidden_dim, mlp_hidden),
                                  nn.Dropout(0.3),
                                  nn.ReLU(True),
                                  nn.Linear(mlp_hidden, 1))

    def forward(self, input):
        entity_value = input[:, :, :, 1]
        entity_type = input[:, :, :, 0]
        relation_value = input[:, :, :, 2]

        ev_embed = self.ev_embedder(entity_value)
        et_embed = self.et_embedder(entity_type)
        rf_embed = self.rf_embedder(relation_value)
        rb_embed = self.rb_embedder(relation_value)

        seq_len = input.size(2)

        rnn_f_input = torch.cat([ev_embed, et_embed, rf_embed], dim=-1).view(-1, seq_len, self.rnn_input)
        _, (rnn_f_output, _) = self.f_rnn(rnn_f_input)

        rnn_b_input = torch.cat([ev_embed, et_embed, rb_embed], dim=-1).view(-1, seq_len, self.rnn_input)
        _, (rnn_b_output, _) = self.b_rnn(flip(rnn_b_input, 1))

        rnn_output = torch.cat((rnn_f_output, rnn_b_output), dim=-1)
        rnn_output = rnn_output.view(input.size(0), input.size(1), -1)
        return self.proj(rnn_output).exp()

path_rnn = PathRNN(entity_vocab_size=11464,
                   entity_type_size=6,
                   relation_vocab_size=9,
                   entity_vocab_dim=50,
                   entity_type_dim=50,
                   relation_vocab_dim=50,
                   rnn_type='LSTM',
                   mlp_hidden=50,
                   rnn_hidden_dim=50)

if torch.cuda.is_available():
    path_rnn = path_rnn.cuda()

def load_file(file_name):
    train_tensor = load_lua(file_name)
    train_dataset = TensorDataset(train_tensor['data'].long() - 1, train_tensor['labels'])
    return train_dataset

def train_files(num_epoch: int = 20):
    print("Start Training...")
    t0 = time.time()
    for epoch in range(num_epoch):
        global_steps = 0
        step = 0
        ce_loss_ = 0
        reg_loss_ = 0
        random.shuffle(torch_train)
        for i, file in enumerate(torch_train):
            train_dset = load_file(file)
            train_iters = DataLoader(dataset=train_dset,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=64,
                                     drop_last=False)
            optimzer = torch.optim.Adam(path_rnn.parameters(), lr = 1e-3)


            for paths, label in train_iters:

                if torch.cuda.is_available():
                    paths, label = paths.cuda(), label.cuda()

                model_out = path_rnn(paths)

                ce_loss = F.binary_cross_entropy(F.sigmoid(model_out.sum(1).log()) + 1e-64,
                                                 label.float(),
                                                 size_average=False)

                otho = torch.mm(path_rnn.rf_embedder.weight, path_rnn.rb_embedder.weight.t())
                identity = torch.diag(torch.Tensor([1.0] * 9))
                if torch.cuda.is_available():
                    identity = identity.cuda()
                otho -= identity

                reg_loss = torch.trace(torch.mm(otho.t(), otho)).sqrt()

                optimzer.zero_grad()
                (ce_loss.div(paths.size(0)) + reg_loss).backward()
                optimzer.step()

                torch.nn.utils.clip_grad_norm_(path_rnn.parameters(), 5.0)

                ce_loss_ += ce_loss.data.item()
                reg_loss_ += reg_loss.data.item()
                step += 1
                global_steps += 1

                print('Epoch [%d/%d] File [%d/%d] Iters %d, CE Loss %.4f Reg Loss %.4f' % (epoch + 1,
                                                                                           num_epoch,
                                                                                           i + 1,
                                                                                           len(torch_train),
                                                                                           global_steps,
                                                                                           ce_loss_ / step,
                                                                                           reg_loss_ / step), end='\r', flush=True)
        print("Duration: {:.4f}\n".format(time.time() - t0))
        if not os.path.isdir('trained_model'):
            os.mkdir('trained_model')

        torch.save(path_rnn, os.path.join('trained_model', 'path_rnn_{}.pt'.format(epoch)))

    print('Start Testing...')
    for i, file in enumerate(torch_test):
        test_dset = load_file(file)
        test_iters = DataLoader(dataset=test_dset,
                                batch_size=64,
                                shuffle=False,
                                num_workers=64,
                                drop_last=False)

        scores_ = []
        for paths, label in test_iters:

            if torch.cuda.is_available():
                paths, label = paths.cuda(), label.cuda()

            model_out = path_rnn(paths)
            scores = F.sigmoid(model_out.sum(1).log())
            scores_.append(list(scores.cpu().data.numpy()))

        if not os.path.isdir('predicted_scores'):
            os.mkdir('predicted_scores')

        with open(os.path.join('predicted_score', file), 'wb') as f:
            cPickle.dump(scores_, f)

if __name__ == "__main__":
    train_files()