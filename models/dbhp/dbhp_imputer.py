import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter

from models.dbhp.utils import deriv_accum_pred
from models.utils import calc_coherence_loss, nll_gauss, reshape_tensor, sample_gauss
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
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TrainablePositionEmbedding(nn.Module):
    MODE_EXPAND = "MODE_EXPAND"
    MODE_ADD = "MODE_ADD"
    MODE_CONCAT = "MODE_CONCAT"

    def __init__(self, num_embeddings, embedding_dim, mode=MODE_ADD):
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
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
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
        raise NotImplementedError("Unknown mode: %s" % self.mode)

    def extra_repr(self):
        return "num_embeddings={}, embedding_dim={}, mode={}".format(
            self.num_embeddings,
            self.embedding_dim,
            self.mode,
        )


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class DBHPImputer(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        self.params = params

        self.dataset = params["dataset"]
        self.n_features = params["n_features"]  # number of features per player
        self.team_size = params["team_size"]  # number of players per team
        self.n_players = self.team_size if self.dataset == "afootball" else self.team_size * 2

        self.pe_z_dim = params["pe_z_dim"]
        self.pi_z_dim = params["pi_z_dim"]
        self.rnn_dim = params["rnn_dim"]
        # self.stochastic = params["stochastic"]

        n_heads = params["n_heads"]
        n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        if params["ppe"] or params["fpe"] or params["fpi"]:
            dp_rnn_input_dim = self.n_features
            if params["ppe"]:
                self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                dp_rnn_input_dim += self.pe_z_dim
            if params["fpe"]:
                self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                dp_rnn_input_dim += self.pe_z_dim
            if params["fpi"]:
                self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
                dp_rnn_input_dim += self.pi_z_dim

        # self.in_fc = nn.Sequential(nn.Linear(rnn_input_dim, self.rnn_dim), nn.ReLU())
        if params["transformer"]:
            # self.pos_encoder = PositionalEncoding(self.rnn_dim, dropout)
            # transformer_encoder_layers = TransformerEncoderLayer(self.rnn_dim, n_heads, self.rnn_dim * 2, dropout)
            self.pos_encoder = PositionalEncoding(dp_rnn_input_dim, dropout)
            transformer_encoder_layers = TransformerEncoderLayer(dp_rnn_input_dim, n_heads, self.rnn_dim * 2, dropout)
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layers, n_layers)
        else:
            self.dp_rnn = nn.LSTM(
                input_size=dp_rnn_input_dim,  # self.rnn_dim,
                hidden_size=self.rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=True,
            )

        dp_fc_dim = self.rnn_dim if params["transformer"] else self.rnn_dim * 2
        self.dp_fc = nn.Linear(dp_fc_dim, self.n_features)

        if self.params["dynamic_hybrid"]:
            self.temp_decay = TemporalDecay(input_size=1, output_size=1, diag=False)
            self.hybrid_rnn_dim = params["hybrid_rnn_dim"]
            hybrid_rnn_input_dim = self.pe_z_dim + self.pi_z_dim + 12

            self.hybrid_rnn = nn.LSTM(
                input_size=hybrid_rnn_input_dim,
                hidden_size=self.hybrid_rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=True,
            )
            self.hybrid_fc = nn.Sequential(
                nn.Linear(self.hybrid_rnn_dim * 2, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1),
            )

    def dynamic_hybrid_pred(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        # dp_pos = data["pos_pred"].transpose(1, 2)  # [bs, time, players, 2]
        # dp_vel = data["vel_pred"].transpose(1, 2)
        # dp_accel = data["cartesian_accel_pred"].transpose(1, 2)
        # dap_f = data["dap_f"].transpose(1, 2)
        # dap_b = data["dap_b"].transpose(1, 2)

        dp_pos = data["pos_pred"].permute(2, 0, 1, 3)  # [time, bs, players, 2]
        dp_vel = data["vel_pred"].permute(2, 0, 1, 3)
        dp_accel = data["cartesian_accel_pred"].permute(2, 0, 1, 3)
        dap_f = data["dap_f"].permute(2, 0, 1, 3)
        dap_b = data["dap_b"].permute(2, 0, 1, 3)
        preds = torch.cat([dp_pos, dp_vel, dp_accel, dap_f, dap_b], dim=-1)  # [time, bs, players, 10]

        z = torch.cat([self.fpe_z, self.fpi_z], -1).reshape(self.seq_len, self.bs, self.n_players, -1)
        gamma_f = self.temp_decay(data["deltas_f"].unsqueeze(-1)).transpose(0, 1)  # [time, bs, players, 1]
        gamma_b = self.temp_decay(data["deltas_b"].unsqueeze(-1)).transpose(0, 1)  # [time, bs, players, 1]

        rnn_input = torch.cat([preds, z, gamma_f, gamma_b], dim=-1).flatten(1, 2)  # [time, bs * players, -1]
        out, _ = self.hybrid_rnn(rnn_input)  # [time, bs * players, hrnn * 2]
        lambdas = self.hybrid_fc(out).unsqueeze(-1)  # [time, bs * players, 3, 1]

        # dp_pos_ = dp_pos.transpose(1, 2).flatten(0, 1).unsqueeze(2)  # [bs * players, time, 1, 2]
        # dap_f_ = dap_f.transpose(1, 2).flatten(0, 1).unsqueeze(2)
        # dap_b_ = dap_b.transpose(1, 2).flatten(0, 1).unsqueeze(2)

        dp_pos_ = dp_pos.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]
        dap_f_ = dap_f.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]
        dap_b_ = dap_b.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]

        preds_pos = torch.cat([dp_pos_, dap_f_, dap_b_], dim=2)  # [time, bs * players, 3, 2]
        hybrid_pos = torch.sum(lambdas * preds_pos, dim=2)  # [time, bs * players, 2]

        # mask = data["pos_mask"].transpose(1, 2)
        # target_pos = data["pos_target"].transpose(1, 2)
        # mask = mask.transpose(1, 2).flatten(0, 1)
        # target_pos = target_pos.transpose(1, 2).flatten(0, 1)

        mask = data["pos_mask"]  # [bs, players, time, 2]
        target_pos = data["pos_target"]
        hybrid_pos = hybrid_pos.reshape(self.seq_len, self.bs, self.n_players, 2).permute(1, 2, 0, 3)
        hybrid_pos = mask * target_pos + (1 - mask) * hybrid_pos  # [bs, players, time, 2]
        # final_out_reshaped = hybrid_pos.reshape(self.bs, self.n_players, self.seq_len, -1)  # [bs, players, time, 2]

        lambdas = lambdas.reshape(self.seq_len, self.bs, self.n_players, 3).permute(1, 2, 0, 3)
        # lambdas_reshaped = lambdas.reshape(self.bs, self.n_players, self.seq_len, -1)
        # [bs, players, time, 3]

        return hybrid_pos, lambdas  # [bs, players, time, 3], [bs, players, time, 2]

    def forward(self, ret: Dict[str, torch.Tensor], device="cuda:0") -> Dict[str, torch.Tensor]:
        # ret = {}
        total_loss = 0.0
        if not self.params["transformer"]:
            self.dp_rnn.flatten_parameters()

        if self.params["cuda"]:
            input = ret["input"].to(device)
            target = ret["target"].to(device)
            mask = ret["mask"].to(device)
            # deltas_f = ret["deltas_f"].to(device)
            # deltas_b = ret["deltas_b"].to(device)

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        self.seq_len, self.bs = input.shape[:2]

        team1_x = input[..., : self.n_features * self.team_size].reshape(-1, self.team_size, self.n_features)
        if self.dataset in ["soccer", "basketball"]:
            team2_x = input[..., self.n_features * self.team_size :].reshape(-1, self.team_size, self.n_features)
            self.x = torch.cat([team1_x, team2_x], 1)  # [time * bs, players, feats]
        else:
            self.x = team1_x  # [time * bs, players, feats]

        rnn_input_list = [self.x]
        if self.params["ppe"]:
            team1_z = self.ppe_st(team1_x)  # [time * bs, team_size, pe_z]
            if self.dataset in ["soccer", "basketball"]:
                team2_z = self.ppe_st(team2_x)  # [time * bs, team_size, pe_z]
                self.ppe_z = torch.cat([team1_z, team2_z], dim=1)  # [time * bs, players, pe_z]
            else:
                self.ppe_z = team1_z  # [time * bs, players, pe_z]
            rnn_input_list += [self.ppe_z]
        if self.params["fpe"]:
            self.fpe_z = self.fpe_st(self.x)  # [time * bs, players, pe_z]
            rnn_input_list += [self.fpe_z]
        if self.params["fpi"]:
            self.fpi_z = self.fpi_st(self.x).unsqueeze(1).expand(-1, self.n_players, -1)  # [time * bs, players, pi_z]
            rnn_input_list += [self.fpi_z]

        rnn_input = torch.cat(rnn_input_list, -1).reshape(self.seq_len, self.bs * self.n_players, -1)
        # rnn_input = self.in_fc(rnn_input)  # [time, bs * players, rnn]

        if self.params["transformer"]:
            rnn_input = self.pos_encoder(rnn_input)
            out = self.transformer_encoder(rnn_input)  # [time, bs * players, rnn * 2]
        else:
            out, _ = self.dp_rnn(rnn_input)  # [time, bs * players, rnn * 2]

        # if self.stochastic:
        #     mean = self.mean_fc(h) # [time, bs * players, out]
        #     std = self.std_fc(h)
        #     out = sample_gauss(mean, std)
        #     mean = mean.reshape(self.seq_len, self.batch_size, -1).transpose(1, 0) # [bs, time, x]
        #     std = std.reshape(self.seq_len, self.batch_size, -1).transpose(1, 0)
        # else:
        #     out = self.out_fc(h) # [time, bs * players, out]

        out = self.dp_fc(out).reshape(self.seq_len, self.bs, -1).transpose(1, 0)  # [bs, time, x]
        pred = mask * target + (1 - mask) * out  # [bs, time, x], STRNN-DP

        # if self.n_features == 2:
        #     pred_ = reshape_tensor(pred, dataset=self.dataset)
        #     target_ = reshape_tensor(target, dataset=self.dataset)
        #     mask_ = reshape_tensor(mask, dataset=self.dataset)
        #     total_loss += torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1 - mask_).sum()
        # else:  # e.g. n_features == 6
        #     feature_types = ["pos", "vel", "cartesian_accel"]

        feature_types = ["pos", "vel", "cartesian_accel"][: self.n_features // 2]

        for mode in feature_types:
            pred_ = reshape_tensor(pred, mode=mode, dataset_type=self.dataset).transpose(1, 2)  # [bs, players, time, 2]
            target_ = reshape_tensor(target, mode=mode, dataset_type=self.dataset).transpose(1, 2)
            mask_ = reshape_tensor(mask, mode=mode, dataset_type=self.dataset).transpose(1, 2)

            # if self.stochastic:
            #     mean_ = reshape_tensor(mean, mode=mode, dataset=self.dataset) # [bs, time, players, 2]
            #     std_ = reshape_tensor(std, mode=mode, dataset=self.dataset)
            #     loss = nll_gauss(mean_, std_, target_)
            # else:
            #     loss = torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1-mask_).sum()

            loss = torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1 - mask_).sum()
            ret[f"{mode}_loss"] = loss
            total_loss += loss

            if mode in ["pos", "vel", "accel", "cartesian_accel"]:
                ret[f"{mode}_pred"] = pred_
                ret[f"{mode}_target"] = target_
                ret[f"{mode}_mask"] = mask_

        ret["accel_loss"] = ret["cartesian_accel_loss"]

        if self.params["deriv_accum"]:
            # DAP-F and DAP-B with output sizes [bs, players, time, 2]
            use_accel = self.params["cartesian_accel"]
            ret["dap_f"] = deriv_accum_pred(ret, use_accel=use_accel, fb="f", dataset=self.dataset)
            ret["dap_b"] = deriv_accum_pred(ret, use_accel=use_accel, fb="b", dataset=self.dataset)

            ret["dap_f_loss"] = torch.abs(ret["pos_target"] - ret["dap_f"]).sum() / (1 - ret["pos_mask"]).sum()
            ret["dap_b_loss"] = torch.abs(ret["pos_target"] - ret["dap_b"]).sum() / (1 - ret["pos_mask"]).sum()

            total_loss += ret["dap_f_loss"] + ret["dap_b_loss"]

            if self.params["dynamic_hybrid"]:
                hybrid_pos, lambdas = self.dynamic_hybrid_pred(ret)
                ret["hybrid_d"] = hybrid_pos  # [bs, players, time, 2] (STRNN-DBHP-D)
                ret["lambdas"] = lambdas  # [bs, players, time, 3]

                sum_loss = torch.abs((ret["hybrid_d"] - ret["pos_target"]) * (1 - ret["pos_mask"])).sum()
                ret["hybrid_d_loss"] = sum_loss / (1 - ret["pos_mask"]).sum()
                total_loss += ret["hybrid_d_loss"]

            if self.params["coherence_loss"]:
                p_d = ret["pos_pred"]  # [bs, time, players, 2]
                v_d = ret["vel_pred"]
                a_d = ret["cartesian_accel_pred"]
                m = ret["pos_mask"][..., 0].unsqueeze(-1)  # [bs, time, players, 1]
                ret["coherence_loss"] = calc_coherence_loss(p_d, v_d, a_d, m, add_va=True)
                total_loss += ret["coherence_loss"]

            # Reshape Predictions
            ret["dap_f"] = ret["dap_f"].transpose(1, 2).flatten(2, 3)  # [bs, time, x]
            ret["dap_b"] = ret["dap_b"].transpose(1, 2).flatten(2, 3)
            ret["hybrid_d"] = ret["hybrid_d"].transpose(1, 2).flatten(2, 3)
            ret["lambdas"] = ret["lambdas"].transpose(1, 2).flatten(2, 3)  # [bs, time, players * 3]

        ret["pred"] = pred
        ret["total_loss"] = total_loss

        return ret
