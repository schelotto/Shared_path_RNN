import torch
import torch.nn as nn

from torch.autograd import Variable
from overrides import overrides

class SharedPathRNN(nn.Module):
    def __init__(self, args):
        super(SharedPathRNN, self).__init__()
        self.r_embed_dim = args.r_embed_dim
        self.e_embed_dim = args.e_embed_dim
        self.rel_size = args.rel_size
        self.ent_size = args.ent_size
        self.rnn_type = args.rnn_type

        self.r_padding = args.r_padding
        self.e_padding = args.e_padding

        self.e_embedding = nn.Embedding(self.ent_size,
                                        self.e_embed_dim,
                                        self)
        self.r_embedding = nn.Embedding(self.rel_size,
                                        self.r_embed_dim)

        self.null = Variable(torch.zeros(self.r_embed_dim)).view(1, 1, -1)
        if torch.cuda.is_available():
            self.null = self.null.cuda()

        self.W_eh = nn.Linear(self.e_embed_dim, self.r_embed_dim, bias=False)
        self.W_rh = nn.Linear(self.r_embed_dim, self.r_embed_dim, bias=False)

        if self.rnn_type in {'RNN', 'LSTM', 'GRU'}:
            self.RNN = getattr(nn, self.rnn_type)(input_size=self.r_embed_dim, bias=False, num_layers=1, hidden_size=self.r_embed_dim)
        else:
            self.RNN = nn.RNN(input_size=self.r_embed_dim, bias=False, num_layers=1, hidden_size=self.r_embed_dim)

    @overrides
    def forward(self,
                entities: torch.LongTensor,  # [e1, ..., en] : [batch, ent_n]
                relations: torch.LongTensor): # [s1, ..., sm] : [batch, rel_m]

        assert entities.size()[-1] == relations.size()[-1] - 1, "size entity list should match relation list"

        if torch.cuda.is_available():
            entities, relations = entities.cuda(), relations.cuda()

        ent_embed = self.e_embedding(entities) # [batch, len_ent, e_embed]
        rel_embed = self.r_embedding(relations) # [batch, len_ent - 1, r_embed]

        null_to_cat = self.null.repeat(relations.size()[0], 1, 1)
        rel_embed = self.concat([rel_embed, null_to_cat], dim=1)

        ent_proj = self.W_eh(ent_embed)
        rel_proj = self.W_rh(rel_embed)

        rnn_out, _ = self.RNN(ent_proj + rel_proj)
        return rnn_out

    def sim_score(self,
                  entities: torch.LongTensor,
                  relations: torch.LongTensor, # [r_embed_dim]
                  relation: torch.FloatTensor): # [batch, r_embed_dim]
        h_t = self.forward(entities, relations) # [batch, r_embed_dim]
        return torch.bmm(relation.unsqueeze(1), h_t.unsqueeze(2))