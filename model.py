import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPathRNN(nn.Module):
    def __init__(self, args):
        super(SharedPathRNN, self).__init__()
        self.r_embed_dim = args.r_embed_dim
        self.e_embed_dim = args.e_embed_dim
        self.rel_size = args.rel_size
        self.ent_size = args.ent_size

        self.r_padding = args.r_padding
        self.e_padding = args.e_padding

        self.e_embedding = nn.Embedding(self.ent_size,
                                        self.e_embed_dim,
                                        self)
        self.r_embedding = nn.Embedding(self.rel_size,
                                        self.r_embed_dim)

        self.W_hh = nn.Linear(self.hidden_size, self.r_embed_dim, bias=False)
        self.W_eh = nn.Linear(self.e_embed_dim, self.r_embed_dim, bias=False)
        self.W_rh = nn.Linear(self.r_embed_dim, self.r_embed_dim, bias=False)

    def forward(self,
                entities: torch.LongTensor,  # [e1, ..., en] : [batch, ent_n]
                relations: torch.LongTensor): # [s1, ..., sm] : [batch, rel_m]

        assert entities.size()[-1] == relations.size()[-1] - 1, "size entity list should match relation list"
        ent_embed = self.e_embedding(entities) # [batch, len_ent, e_embed]
        rel_embed = self.r_embedding(relations) # [batch, len_rel, r_embed]

        for i in range(ent_embed.size()[1]):
            if i == 0:
                h_t = F.sigmoid(self.W_eh(ent_embed[:, i, :]))
            else:
                h_t = F.sigmoid(self.W_hh(h_t) + self.W_en(ent_embed[:, i, :]) + self.W_rh(rel_embed[:, i-1, :]))

        return h_t

    def sim_score(self,
            entities: torch.LongTensor,
            relations: torch.LongTensor,
            relation: torch.FloatTensor): # [batch, r_embed_dim]
        h_t = self.forward(entities, relations) # [batch, r_embed_dim]
        return torch.bmm(relation.unsqueeze(1), h_t.unsqueeze(2))