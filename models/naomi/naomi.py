import random

import numpy as np
import torch
import torch.nn as nn

from models.naomi.naomi_imputer import NAOMIImputer
from models.naomi.naomi_imputer_alpha import NAOMIImputerAlpha
from models.naomi.utils import *
from models.utils import *


class NAOMI(nn.Module):
    def __init__(self, params, parser=None):
        super(NAOMI, self).__init__()
        self.model_args = [
            "missing_pattern",
            "n_players",
            "rnn_dim",
            "dec1_dim",
            "dec2_dim",
            "dec4_dim",
            "dec8_dim",
            "dec16_dim",
            "n_layers",
            "n_highest",
            "cartesian_accel",
            "xy_sort",
            "stochastic",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        if self.params["missing_pattern"] == "playerwise":
            self.model = NAOMIImputerAlpha(self.params)  # not implemented yet.
        else:
            self.model = NAOMIImputer(self.params)

    def forward(self, data, mode="train", teacher_forcing=False, device="cuda:0"):
        ret = {"loss": 0, "pos_dist": 0}

        n_features = self.params["n_features"]
        dataset = self.params["dataset"]

        if dataset == "soccer":
            total_players = 22
        elif dataset == "basketball":
            total_players = 10
        else:  # e.g. "football"
            total_players = 6

        if self.params["xy_sort"]:
            input_data, sort_idxs = xy_sort_tensor(data[0], n_players=total_players)  # [bs, time, x]
            target_data = input_data.clone()
        else:
            if dataset == "football":  # randomly permute player order for NFL dataset.
                data[0] = random_permutation(data[0], total_players)
                data[1] = data[0].clone()
            input_data = data[0]  # [bs, time, x]
            target_data = data[1]

        bs, seq_len = input_data.shape[0], input_data.shape[1]

        missing_probs = np.arange(10) * 0.1
        mask = generate_mask(
            data_dict=ret,
            mode=self.params["missing_pattern"],
            window_size=seq_len,
            missing_rate=missing_probs[random.randint(1, 9)],
            sports=dataset,
        )
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = torch.repeat_interleave(mask, n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x]

        if self.params["cuda"]:
            input_data, target_data, mask = input_data.to(device), target_data.to(device), mask.to(device)

        masked_input = input_data * mask

        masked_input = masked_input.transpose(0, 1)  # [bs, time, x] to [time, bs, x]
        target_data = target_data.transpose(0, 1)
        mask = mask.transpose(0, 1)

        if self.params["missing_pattern"] == "playerwise":
            masked_input = masked_input.reshape(seq_len, bs, total_players, -1)  # [time, bs, 22, feat_dim]
            target_data = target_data.reshape(seq_len, bs, total_players, -1)
            mask = mask.reshape(seq_len, bs, total_players, -1)

            has_value = torch.ones_like(mask, dtype=torch.float32)  # [time, bs, 22, feat_dim]
            has_value = has_value * mask
            has_value = has_value[..., 0, None]  # [time, bs, 22, 1]

            input_data = torch.cat([has_value, masked_input], dim=-1)  # [time, bs, 22, 1+feat_dim]

        elif self.params["missing_pattern"] == "uniform":  # e.g. block all features.
            has_value = torch.ones(seq_len, bs, 1)
            if self.params["cuda"]:
                has_value = has_value.to(device)
            has_value = has_value * mask[..., 0, None]  # [time, bs, 1]
            input_data = torch.cat([has_value, masked_input], dim=-1)  # [time, bs, 1+feat_dim]

        if teacher_forcing:
            batch_loss, pos_dist = self.model(input_data, target_data)
            ret["total_loss"] = batch_loss
            ret["pred_dist"] = pos_dist
        else:
            data_list = []
            for j in range(seq_len):
                data_list.append(input_data[j : j + 1])
            pred = self.model.sample(data_list)  # [time, bs, feat_dim]

            pred = pred.transpose(0, 1)  # [bs, time, feat_dim]
            if self.params["missing_pattern"] == "playerwise":
                target_ = target_data.flatten(2, 3).transpose(0, 1)
                mask_ = mask.flatten(2, 3).transpose(0, 1)
            else:
                target_ = target_data.transpose(0, 1)
                mask_ = mask.transpose(0, 1)

            batch_loss = self.model.calc_mae_loss(pred, target_)

            aggfunc = "mean" if mode == "train" else "sum"
            pos_dist = calc_trace_dist(pred, target_, mask_, n_features=n_features, aggfunc=aggfunc, dataset=dataset)

            ret["total_loss"] = batch_loss
            ret["pred_dist"] = pos_dist

            ret["pred"] = pred
            ret["target"] = target_
            ret["mask"] = mask_

            if self.params["xy_sort"]:
                ret["pred"] = xy_sort_tensor(ret["pred"], sort_idxs, total_players, mode="restore")
                ret["target"] = xy_sort_tensor(ret["target"], sort_idxs, total_players, mode="restore")
                ret["mask"] = xy_sort_tensor(ret["mask"], sort_idxs, total_players, mode="restore")

        return ret

    def forward2(self, input_dict, device="cuda:0"):
        ret = {}

        masked_input = input_dict["input"].transpose(0, 1)  # [time, bs, feat_dim]
        target = input_dict["target"].transpose(0, 1)
        mask = input_dict["mask"].transpose(0, 1)

        seq_len, bs = target.shape[:2]

        has_value = torch.ones(seq_len, bs, 1)
        if self.params["cuda"]:
            has_value = has_value.to(device)
        has_value = has_value * mask[..., 0, None]  # [time, bs, 1]
        input_data = torch.cat([has_value, masked_input], dim=-1)  # [time, bs, 1+feat_dim]

        data_list = []
        for j in range(seq_len):
            data_list.append(input_data[j : j + 1])
        pred = self.model.sample(data_list)  # [time, bs, feat_dim]

        ret["pred"] = pred.transpose(0, 1)  # [bs, time, feat_dim]
        ret["target"] = target.transpose(0, 1)
        ret["mask"] = mask.transpose(0, 1)

        return ret
