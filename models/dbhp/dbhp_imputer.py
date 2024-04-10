import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter

from models.dbhp.utils import derivative_based_pred
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
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.max_len = 200

        n_heads = params["n_heads"]
        n_layers = params["n_layers"]

        self.dataset = params["dataset"]
        self.n_features = params["n_features"]  # number of features per player
        self.n_players = params["n_players"]  # number of agents per team
        self.n_components = self.n_players  # total number of agents

        if self.dataset in ["soccer", "basketball"]:
            self.n_components *= 2

        self.pe_z_dim = params["pe_z_dim"]
        self.pi_z_dim = params["pi_z_dim"]
        self.rnn_dim = params["rnn_dim"]
        # self.stochastic = params["stochastic"]

        self.n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        if params["ppe"] or params["fpe"] or params["fpi"]:
            rnn_input_dim = self.n_features

            if params["ppe"]:
                self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                rnn_input_dim += self.pe_z_dim
            if params["fpe"]:
                self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                rnn_input_dim += self.pe_z_dim
            if params["fpi"]:
                self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
                rnn_input_dim += self.pi_z_dim

        self.in_fc = nn.Sequential(nn.Linear(rnn_input_dim, self.rnn_dim), nn.ReLU())
        if params["transformer"]:
            self.pos_encoder = PositionalEncoding(self.rnn_dim, dropout)
            self.pos_embedder = TrainablePositionEmbedding(
                num_embeddings=self.max_len, embedding_dim=self.rnn_dim, mode=TrainablePositionEmbedding.MODE_ADD
            )
            transformer_encoder_layers = TransformerEncoderLayer(self.rnn_dim, n_heads, self.rnn_dim * 2, dropout)
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layers, n_layers)
        else:  # e.g. Bi-LSTM
            self.rnn = nn.LSTM(
                input_size=self.rnn_dim,
                hidden_size=self.rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=True,
            )

        out_fc_dim = self.rnn_dim if params["transformer"] else self.rnn_dim * 2

        self.temp_decay = TemporalDecay(input_size=1, output_size=1, diag=False)

        if self.params["train_hybrid"]:
            self.hybrid_rnn_dim = params["hybrid_rnn_dim"]
            hybrid_rnn_input_dim = 2 * 6 + self.pe_z_dim + self.pi_z_dim

            self.hybrid_rnn = nn.LSTM(
                input_size=hybrid_rnn_input_dim,
                hidden_size=self.hybrid_rnn_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            self.hybrid_out_fc = nn.Sequential(nn.Linear(self.hybrid_rnn_dim * 2, 3), nn.Softmax(dim=-1))

        self.out_fc = nn.Linear(out_fc_dim, self.n_features)

    def train_hybrid_pred(self, ret):
        xy_pred = ret["xy_pred"].transpose(1, 2)  # [bs, time, comp, 2]
        physics_f_pred = ret["physics_f_pred"].transpose(1, 2)
        physics_b_pred = ret["physics_b_pred"].transpose(1, 2)
        vel_pred = ret["vel_pred"].transpose(1, 2)
        accel_pred = ret["cartesian_accel_pred"].transpose(1, 2)
        xy_mask = ret["xy_mask"].transpose(1, 2)
        xy_target = ret["xy_target"].transpose(1, 2)

        gamma_f = self.temp_decay(ret["deltas_f"].unsqueeze(-1))  # [bs, time, comp, 1]
        gamma_b = self.temp_decay(ret["deltas_b"].unsqueeze(-1))

        input_preds = torch.cat([xy_pred, physics_f_pred, physics_b_pred], dim=-1)  # [bs, time, comp, 2 * 3]
        contexts = self.hybrid_rnn_context  # [bs, time, comp, -1]
        hybrid_rnn_inputs = torch.cat([input_preds, contexts, gamma_f, gamma_b, vel_pred, accel_pred], dim=-1)
        hybrid_rnn_inputs = hybrid_rnn_inputs.transpose(1, 2).flatten(0, 1)  # [bs * comp, time, -1]

        hybrid_out, _ = self.hybrid_rnn(hybrid_rnn_inputs)  # [bs * comp, time, rnn_dim * 2]
        hybrid_weights = self.hybrid_out_fc(hybrid_out)  # [bs * comp, time, 3]

        xy_pred_ = xy_pred.transpose(1, 2).flatten(0, 1).unsqueeze(2)  # [bs * comp, time, 1, 2]
        physics_f_pred_ = physics_f_pred.transpose(1, 2).flatten(0, 1).unsqueeze(2)
        physics_b_pred_ = physics_b_pred.transpose(1, 2).flatten(0, 1).unsqueeze(2)
        xy_mask_ = xy_mask.transpose(1, 2).flatten(0, 1)
        xy_target_ = xy_target.transpose(1, 2).flatten(0, 1)

        preds_ = torch.cat([xy_pred_, physics_f_pred_, physics_b_pred_], dim=2)  # [bs * comp, time, 3, 2]
        hybrid_weights_ = hybrid_weights.unsqueeze(-1)  # [bs * comp, time, 3, 1]
        hybrid_pred = torch.sum(preds_ * hybrid_weights_, dim=2)  # [bs * comp, time, 2]

        final_out = xy_mask_ * xy_target_ + (1 - xy_mask_) * hybrid_pred  # [bs * comp, time, 2]
        final_out_reshaped = final_out.reshape(self.bs, self.n_components, self.seq_len, -1)  # [bs, comp, time, 2]

        hybrid_weights_reshaped = hybrid_weights.reshape(
            self.bs, self.n_components, self.seq_len, -1
        )  # [bs, comp, time, 3]

        return final_out_reshaped, hybrid_weights_reshaped

    def forward(self, data, device="cuda:0"):
        ret = {}
        total_loss = 0.0
        if not self.params["transformer"]:
            self.rnn.flatten_parameters()

        if self.params["cuda"]:
            input = data["input"].to(device)
            target = data["target"].to(device)
            mask = data["mask"].to(device)
            deltas_f = data["deltas_f"].to(device)
            deltas_b = data["deltas_b"].to(device)

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]

        self.seq_len, self.bs = input.shape[:2]

        team1_x = input[..., : self.n_features * self.n_players].reshape(-1, self.n_players, self.n_features)
        if self.dataset in ["soccer", "basketball"]:
            team2_x = input[..., self.n_features * self.n_players :].reshape(-1, self.n_players, self.n_features)
            self.x = torch.cat([team1_x, team2_x], 1)  # [time * bs, n_agents, n_features * 2]
        else:
            self.x = team1_x  # [time * bs, n_agents, n_features]

        rnn_input_list = [self.x]
        if self.params["ppe"]:
            team1_z = self.ppe_st(team1_x)  # [time * bs, team1_agents, pe_z_dim]
            if self.dataset in ["soccer", "basketball"]:
                team2_z = self.ppe_st(team2_x)
                self.ppe_z = torch.cat([team1_z, team2_z], dim=1)  # [time * bs, n_agents, pe_z_dim]
            else:
                self.ppe_z = team1_z
            rnn_input_list += [self.ppe_z]
        if self.params["fpe"]:
            self.fpe_z = self.fpe_st(self.x)  # [time * bs, n_agents, pe_z_dim]
            rnn_input_list += [self.fpe_z]
        if self.params["fpi"]:
            self.fpi_z = (
                self.fpi_st(self.x).unsqueeze(1).expand(-1, self.n_components, -1)
            )  # [time * bs, n_agents, pi_z_dim]
            rnn_input_list += [self.fpi_z]

        contexts = torch.cat(rnn_input_list, -1).reshape(self.seq_len, self.bs * self.n_components, -1)
        rnn_input = self.in_fc(contexts)  # [time, bs * n_agents, rnn_dim]

        if self.params["train_hybrid"]:
            context_embeds = torch.cat([self.fpe_z, self.fpi_z], -1).reshape(
                self.seq_len, self.bs * self.n_components, -1
            )
            self.hybrid_rnn_context = context_embeds.reshape(self.seq_len, self.bs, self.n_components, -1).transpose(
                0, 1
            )  # [time, bs * comp, -1]

        if self.params["transformer"]:
            rnn_input = self.pos_encoder(rnn_input)
            out = self.transformer_encoder(rnn_input)  # [time, bs * n_agents, -1]
        else:
            out, _ = self.rnn(rnn_input)

        # if self.stochastic:
        #     mean = self.mean_fc(h) # [time, bs * comp, out_dim]
        #     std = self.std_fc(h)
        #     out = sample_gauss(mean, std)
        #     mean = mean.reshape(self.seq_len, self.batch_size, -1).transpose(1, 0) # [bs, time, x_dim]
        #     std = std.reshape(self.seq_len, self.batch_size, -1).transpose(1, 0)
        # else:
        #     out = self.out_fc(h) # [time, bs * comp, out_dim]
        out = self.out_fc(out)  # [time, bs * n_agents, feat_dim]
        out = out.reshape(self.seq_len, self.bs, -1).transpose(1, 0)  # [bs, time, x_dim] (STRNN-DP)

        pred = mask * target + (1 - mask) * out  # [bs, time, x_dim]

        if self.n_features == 2:
            pred_ = reshape_tensor(pred, dataset=self.dataset)
            target_ = reshape_tensor(target, dataset=self.dataset)
            mask_ = reshape_tensor(mask, dataset=self.dataset)
            total_loss += torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1 - mask_).sum()
        else:  # e.g. n_features == 6
            feature_types = ["xy", "vel", "cartesian_accel"]

            for mode in feature_types:
                pred_ = reshape_tensor(pred, mode=mode, dataset=self.dataset).transpose(1, 2)  # [bs, comp, time, 2]
                target_ = reshape_tensor(target, mode=mode, dataset=self.dataset).transpose(1, 2)
                mask_ = reshape_tensor(mask, mode=mode, dataset=self.dataset).transpose(1, 2)
                # if self.stochastic:
                #     mean_ = reshape_tensor(mean, mode=mode, dataset=self.dataset) # [bs, time, comp, 2]
                #     std_ = reshape_tensor(std, mode=mode, dataset=self.dataset)
                #     loss = nll_gauss(mean_, std_, target_)
                # else:
                #     loss = torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1-mask_).sum()

                loss = torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1 - mask_).sum()

                if mode in ["xy"]:
                    ret[f"{mode}_loss"] = loss
                    total_loss += loss
                else:
                    ret[f"{mode}_loss"] = loss
                    total_loss += loss

                if mode in ["xy", "vel", "accel", "cartesian_accel"]:
                    ret[f"{mode}_pred"] = pred_
                    ret[f"{mode}_target"] = target_
                    ret[f"{mode}_mask"] = mask_

            ret["deltas_f"] = deltas_f
            ret["deltas_b"] = deltas_b

            ret["accel_loss"] = ret["cartesian_accel_loss"]

            if self.params["physics_loss"]:
                mode = "accel" if self.params["cartesian_accel"] else "vel"
                ret["physics_f_pred"] = derivative_based_pred(
                    ret, physics_mode=mode, fb="f", dataset=self.dataset
                )  # [bs, n_agents, time, 2] (STRNN-DAP-F)
                ret["physics_b_pred"] = derivative_based_pred(
                    ret, physics_mode=mode, fb="b", dataset=self.dataset
                )  # (STRNN-DAP-B)

                ret["physics_f_loss"] = (
                    torch.abs(ret["xy_target"] - ret["physics_f_pred"]).sum() / (1 - ret["xy_mask"]).sum()
                )
                ret["physics_b_loss"] = (
                    torch.abs(ret["xy_target"] - ret["physics_b_pred"]).sum() / (1 - ret["xy_mask"]).sum()
                )

                total_loss += ret["physics_f_loss"] + ret["physics_b_loss"]

                if self.params["train_hybrid"]:
                    hybrid_o, hybrid_w = self.train_hybrid_pred(ret)
                    ret["train_hybrid_pred"] = hybrid_o  # [bs, n_agents, time, 2] (STRNN-DBHP-D)
                    ret["train_hybrid_weights"] = hybrid_w  # [bs, n_agents, time, 3]
                    ret["train_hybrid_loss"] = (
                        torch.abs((ret["train_hybrid_pred"] - ret["xy_target"]) * (1 - ret["xy_mask"])).sum()
                        / (1 - ret["xy_mask"]).sum()
                    )
                    total_loss += ret["train_hybrid_loss"]

                if self.params["coherence_loss"]:
                    p_d = ret["xy_pred"]  # [bs, time, n_agents, 2]
                    v_d = ret["vel_pred"]
                    a_d = ret["cartesian_accel_pred"]
                    m = ret["xy_mask"][..., 0].unsqueeze(-1)  # [bs, time, n_agents, 1]
                    ret["coherence_loss"] = calc_coherence_loss(p_d, v_d, a_d, m, add_va=True)
                    total_loss += ret["coherence_loss"]

                # Reshape Predictions
                ret["physics_f_pred"] = ret["physics_f_pred"].transpose(1, 2).flatten(2, 3)  # [bs, time, x_dim]
                ret["physics_b_pred"] = ret["physics_b_pred"].transpose(1, 2).flatten(2, 3)
                ret["train_hybrid_pred"] = ret["train_hybrid_pred"].transpose(1, 2).flatten(2, 3)
                ret["train_hybrid_weights"] = (
                    ret["train_hybrid_weights"].transpose(1, 2).flatten(2, 3)
                )  # [bs, time, comp*3]

        ret.update(
            {"total_loss": total_loss, "pred": pred, "input": input.transpose(0, 1), "target": target, "mask": mask}
        )

        return ret
