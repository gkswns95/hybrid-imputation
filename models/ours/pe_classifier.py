import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
)

from set_transformer.model import SetTransformer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TrainablePositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(TrainablePositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        x = x.transpose(0, 1)

        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings,self.num_embeddings) + self.num_embeddings
            x = F.embedding(indices.type(torch.LongTensor), self.weight)
            return x.transpose(0, 1)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            x = x + embeddings
            return x.transpose(0, 1)
        if self.mode == self.MODE_CONCAT:
            x = torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
            return x.transpose(0, 1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(self.num_embeddings, self.embedding_dim, self.mode,)

class PeClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.max_len = 200

        n_heads = params["n_heads"]
        n_layers = params["n_layers"]

        self.target_type = params["target_type"]

        self.x_dim = params["n_features"] # number of features per player (6 in general)
        self.n_players = params["n_players"] # number of players per team (11 in general)
        self.n_components = self.n_players * 2 + 4 # number of total players + 4 outside labels (26 in general)

        self.pe_z_dim = params["pe_z_dim"]
        self.pi_z_dim = params["pi_z_dim"]
        self.rnn_dim = params["rnn_dim"]

        self.n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        if params["ppe"] or params["fpe"] or params["fpi"]:
            rnn_input_dim = self.x_dim

            if params["ppe"]:
                self.ppe_st = SetTransformer(self.x_dim, self.pe_z_dim, embed_type="e")
                self.outside_fc = nn.Linear(self.x_dim, self.pe_z_dim)
                rnn_input_dim += self.pe_z_dim
            if params["fpe"]:
                self.fpe_st = SetTransformer(self.x_dim, self.pe_z_dim, embed_type="e")
                rnn_input_dim += self.pe_z_dim
            if params["fpi"]:
                self.fpi_st = SetTransformer(self.x_dim, self.pi_z_dim, embed_type="i")
                rnn_input_dim += self.pi_z_dim

        self.in_fc = nn.Sequential(nn.Linear(rnn_input_dim, self.rnn_dim), nn.ReLU())
        if params["transformer"]:
            self.pos_encoder = PositionalEncoding(self.rnn_dim, dropout)
            self.pos_embedder = TrainablePositionEmbedding(num_embeddings=self.max_len, embedding_dim=self.rnn_dim, mode=TrainablePositionEmbedding.MODE_ADD)
            transformer_encoder_layers = TransformerEncoderLayer(self.rnn_dim, n_heads, self.rnn_dim * 2, dropout)
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layers, n_layers)
        else:  # e.g. Bi-LSTM
            self.rnn = nn.LSTM(
                input_size=self.rnn_dim,
                hidden_size=self.rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=True
            )

        out_fc_dim = self.rnn_dim if params["transformer"] else self.rnn_dim * 2

        self.out_fc = nn.Linear(out_fc_dim, 1)

    def forward(self, data, device="cuda:0"):
        if not self.params["transformer"]:
            self.rnn.flatten_parameters()

        input = data["input"].to(device)
        target = data["target"].to(device)

        input = input.transpose(0, 1) # [bs, time, -1] to [time, bs, -1]
        seq_len = input.size(0)
        batch_size = input.size(1)
        n_players = self.params["n_players"]
    
        team1_x = input[:, :, : self.x_dim * n_players].reshape(-1, n_players, self.x_dim) # [time * bs, player, x]
        team2_x = input[:, :, self.x_dim * n_players : -(self.x_dim * 4)].reshape(-1, n_players, self.x_dim)
        outside_x = input[:, :, -(self.x_dim * 4) :].reshape(-1, 4, self.x_dim)

        self.x = torch.cat([team1_x, team2_x, outside_x], 1)  # [time * bs, comp, x]

        rnn_input_list = [self.x]
        if self.params["ppe"]:
            team1_z = self.ppe_st(team1_x)  # [time * bs, player, pe_z_dim]
            team2_z = self.ppe_st(team2_x) 
            outside_z = self.outside_fc(outside_x) # [time * bs, 4, pe_z_dim]
            self.ppe_z = torch.cat([team1_z, team2_z, outside_z], dim=1) # [time * bs, comp, pe_z_dim]
            rnn_input_list += [self.ppe_z]
        if self.params["fpe"]:
            self.fpe_z = self.fpe_st(self.x)  # [time * bs, comp, pe_z_dim]
            rnn_input_list += [self.fpe_z]
        if self.params["fpi"]:
            self.fpi_z = self.fpi_st(self.x).unsqueeze(1).expand(-1, self.n_components, -1)  # [time * bs, comp, pi_z_dim]
            rnn_input_list += [self.fpi_z]

        contexts = torch.cat(rnn_input_list, -1).reshape(seq_len, batch_size * self.n_components, -1)
        rnn_input = self.in_fc(contexts) # [time, bs * comp, rnn_dim]

        if self.params["transformer"]:
            # rnn_input = self.pos_embedder(rnn_input)
            rnn_input = self.pos_encoder(rnn_input)
            h = self.transformer_encoder(rnn_input) # [time, bs * comp, -1]
        else: # e.g. Bi-LSTM 
            h, _ = self.rnn(rnn_input)

        h = h.reshape(seq_len * batch_size, self.n_components, -1) # [time * bs, comp, rnn_dim]    
        out = self.out_fc(h).squeeze(-1).reshape(seq_len, batch_size, -1) # [time, bs, comp]
        out = out.transpose(0,1).transpose(1,2) # [bs, comp, time]

        ce_loss = nn.CrossEntropyLoss()(out, target)

        return {"loss": ce_loss, "out": torch.softmax(out, dim=1), "input": input.transpose(0, 1), "target": target}