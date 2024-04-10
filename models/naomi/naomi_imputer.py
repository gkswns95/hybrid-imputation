import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.naomi.utils import *
from models.utils import reshape_tensor


def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.n_features = params["n_features"]  # number of features per each player
        self.n_players = params["n_players"]  # number of players per each team

        self.hidden_dim = params["discrim_rnn_dim"]
        if params["dataset"] in ["soccer", "basketball"]:
            y_dim = (self.n_players * 2) * self.n_features
        else:
            y_dim = self.n_players * self.n_features
        self.action_dim = y_dim
        self.state_dim = y_dim
        self.gpu = params["cuda"]
        self.num_layers = params["discrim_num_layers"]

        self.gru = nn.GRU(self.state_dim, self.hidden_dim, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, a, h=None):
        """
        x : [seq, batch, x_dim]
        a : [seq, batch, x_dim]
        """
        p, _ = self.gru(x, h)  # [seq, batch, x_dim]
        p = torch.cat([p, a], 2)  # [seq, batch, x_dim * 2]
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))  # [seq, batch, 1]
        return prob

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))


class NAOMIImputer(nn.Module):
    def __init__(self, params):
        super(NAOMIImputer, self).__init__()
        self.params = params

        self.n_features = params["n_features"]  # number of features per each player
        self.n_players = params["n_players"]  # number of players per each team
        self.stochastic = params["stochastic"]
        self.dataset = params["dataset"]

        if self.dataset in ["soccer", "basketball"]:
            self.y_dim = (self.n_players * 2) * self.n_features
        else:
            self.y_dim = self.n_players * self.n_features

        self.rnn_dim = params["rnn_dim"]
        self.n_layers = params["n_layers"]
        self.highest = params["n_highest"]
        self.batch_size = params["batch_size"]
        self.dims = {}
        self.networks = {}

        self.gru = nn.GRU(self.y_dim, self.rnn_dim, self.n_layers)
        self.back_gru = nn.GRU(self.y_dim + 1, self.rnn_dim, self.n_layers)

        step = 1
        while step <= self.highest:
            k = str(step)
            self.dims[k] = params["dec" + k + "_dim"]
            dim = self.dims[k]

            curr_level = {}
            curr_level["dec"] = nn.Sequential(nn.Linear(2 * self.rnn_dim, dim), nn.ReLU())
            curr_level["mean"] = nn.Linear(dim, self.y_dim)
            if self.stochastic:
                curr_level["std"] = nn.Sequential(nn.Linear(dim, self.y_dim), nn.Softplus())
            curr_level = nn.ModuleDict(curr_level)

            self.networks[k] = curr_level

            step = step * 2

        self.networks = nn.ModuleDict(self.networks)

    def forward(self, data, ground_truth):
        """
        data : [time, bs, 1 + feat_dim]
        ground_truth : [time, bs, feat_dim]
        """

        device = data.device
        seq_len = data.shape[0]
        bs = data.shape[1]

        h = torch.zeros(self.n_layers, bs, self.rnn_dim)
        h_back = torch.zeros(self.n_layers, bs, self.rnn_dim)

        if self.params["cuda"]:
            h, h_back = h.to(device), h_back.to(device)

        loss = 0.0
        pos_dist = 0.0
        count = 0
        imput_count = 0
        h_back_dict = {}

        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t + 1] = h_back
            state_t = data[t]
            _, h_back = self.back_gru(state_t.unsqueeze(0), h_back)

        for t in range(seq_len):
            state_t = ground_truth[t]
            _, h = self.gru(state_t.unsqueeze(0), h)
            count += 1
            for k, dim in self.dims.items():
                step_size = int(k)
                curr_level = self.networks[str(step_size)]
                if t + 2 * step_size <= seq_len:
                    next_t = ground_truth[t + step_size]
                    h_back = h_back_dict[t + 2 * step_size]

                    dec_t = curr_level["dec"](torch.cat([h[-1], h_back[-1]], 1))
                    dec_mean_t = curr_level["mean"](dec_t)  # [bs, y_dim]

                    if self.stochastic:
                        dec_std_t = curr_level["std"](dec_t)  # [bs, y_dim]
                        loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                        dec_mean_t = reparam_sample_gauss(dec_mean_t, dec_std_t)  # [bs, y_dim]
                    else:
                        loss += self.calc_mae_loss(dec_mean_t, next_t)

                    pred_t = reshape_tensor(dec_mean_t, rescale=True, dataset=self.dataset)  # [bs, total_players, 2]
                    target_t = reshape_tensor(next_t, rescale=True, dataset=self.dataset)
                    pos_dist += torch.norm(pred_t - target_t, dim=-1).mean()

                    imput_count += 1

        loss = loss / count / bs
        pos_dist = pos_dist / imput_count

        return loss, pos_dist.item()

    def sample(self, data_list):
        device = data_list[0].device
        bs = data_list[0].shape[1]
        seq_len = len(data_list)

        h = torch.zeros(self.params["n_layers"], bs, self.rnn_dim)
        h_back = torch.zeros(self.params["n_layers"], bs, self.rnn_dim)
        if self.params["cuda"]:
            h, h_back = h.to(device), h_back.to(device)

        h_back_dict = {}
        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t + 1] = h_back
            state_t = data_list[t]  # [1, bs, 1+feat_dim]
            _, h_back = self.back_gru(state_t, h_back)

        curr_p = 0  # pivot 1
        _, h = self.gru(data_list[curr_p][:, :, 1:], h)
        while curr_p < seq_len - 1:
            if data_list[curr_p + 1][0, 0, 0] == 1:  # indicator = 1 (observed_value)
                curr_p += 1
                _, h = self.gru(data_list[curr_p][:, :, 1:], h)
            else:  # indicator = 0 (missing_Value)
                next_p = curr_p + 1  # pivot 2
                while next_p < seq_len and data_list[next_p][0, 0, 0] == 0:
                    next_p += 1

                step_size = 1
                while curr_p + 2 * step_size <= next_p and step_size <= self.highest:
                    step_size *= 2
                step_size = step_size // 2

                self.interpolate(data_list, curr_p, h, h_back_dict, step_size, device=device)

        return torch.cat(data_list, dim=0)[:, :, 1:]  # [time, bs, feat_dim]

    def interpolate(self, data_list, curr_p, h, h_back_dict, step_size, p=0, device="cuda:0"):
        h_back = h_back_dict[curr_p + 2 * step_size]
        curr_level = self.networks[str(step_size)]

        dec_t = curr_level["dec"](torch.cat([h[-1], h_back[-1]], 1))
        dec_mean_t = curr_level["mean"](dec_t)
        if self.stochastic:
            dec_std_t = curr_level["std"](dec_t)
            state_t = reparam_sample_gauss(dec_mean_t, dec_std_t)
        else:
            state_t = dec_mean_t

        added_state = state_t.unsqueeze(0)  # [1, bs, y_dim]
        has_value = torch.ones(added_state.shape[0], added_state.shape[1], 1)
        if self.params["cuda"]:
            has_value = has_value.to(device)
        added_state = torch.cat([has_value, added_state], 2)
        if step_size > 1:
            right = curr_p + step_size
            left = curr_p + step_size // 2
            h_back = h_back_dict[right + 1]
            _, h_back = self.back_gru(added_state, h_back)
            h_back_dict[right] = h_back

            zeros = torch.zeros(added_state.shape[0], added_state.shape[1], self.y_dim + 1)
            if self.params["cuda"]:
                zeros = zeros.to(device)
            for i in range(right - 1, left - 1, -1):
                _, h_back = self.back_gru(zeros, h_back)
                h_back_dict[i] = h_back

        data_list[curr_p + step_size] = added_state

    def calc_mae_loss(self, pred, target):
        """
        pred : [bs, feat_dim]
        target : [bs, feat_dim]
        """
        loss = 0.0

        if self.n_features == 2:
            feature_types = ["xy"]
            scale_factor = 1
        elif self.n_features == 4:
            feature_types = ["xy", "vel"]
            scale_factor = 10
        elif self.n_features == 6:
            if self.params["cartesian_accel"]:
                feature_types = ["xy", "vel", "cartesian_accel"]
            else:
                feature_types = ["xy", "vel", "speed", "accel"]
            scale_factor = 10

        for mode in feature_types:
            pred_ = reshape_tensor(pred, mode=mode, dataset=self.dataset)  # [bs, total_players, -1]
            target_ = reshape_tensor(target, mode=mode, dataset=self.dataset)

            mae_loss = torch.abs(pred_ - target_).mean()

            if mode in ["accel", "speed"]:
                loss += mae_loss * 0
            elif mode in ["xy"]:
                loss += mae_loss * scale_factor
            else:
                loss += mae_loss

        return loss
