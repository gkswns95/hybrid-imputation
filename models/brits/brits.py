import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.brits.rits import RITS
from models.utils import *


class BRITS(nn.Module):
    def __init__(self, params, parser=None):
        super(BRITS, self).__init__()

        self.model_args = [
            "missing_pattern",
            "n_players",
            "rnn_dim",
            "dropout",
            "cartesian_accel",
            "xy_sort",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        self.rits_f = RITS(self.params)
        self.rits_b = RITS(self.params)

    def forward(self, data, mode="train", device="cuda:0"):
        total_players = self.params["n_players"]
        if self.params["dataset"] in ["soccer", "basketball"]:
            total_players *= 2

        n_features = self.params["n_features"]
        dataset = self.params["dataset"]

        if self.params["xy_sort"]:
            input_data, sort_idxs = xy_sort_tensor(data[0], n_players=total_players)  # [bs, time, x_dim]
            target_data = input_data.clone()
        else:
            if dataset == "football":  # Randomly permute player order for NFL dataset.
                data[0] = random_permutation(data[0], total_players)
                data[1] = data[0].clone()
            input_data = data[0]  # [bs, time, x_dim]
            target_data = data[1]

        if self.params["dataset"] == "soccer":
            ball_data = data[2]
        else:
            ball_data = data[1]

        input_dict = {"target": target_data, "ball": ball_data}

        bs, seq_len = input_dict["target"].shape[:2]

        missing_probs = np.arange(10) * 0.1
        mask = generate_mask(
            inputs=input_dict,
            mode=self.params["missing_pattern"],
            ws=seq_len,
            missing_rate=missing_probs[random.randint(1, 9)],
            dataset=dataset,
        )

        if self.params["missing_pattern"] == "camera_simulate":
            time_gap = time_interval(mask, list(range(seq_len)), mode="camera")
            mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, n_players]
            mask = torch.repeat_interleave(mask, n_features, dim=-1)
            time_gap = torch.repeat_interleave(time_gap, n_features, dim=-1)
        else:
            time_gap = time_interval(mask, list(range(seq_len)), mode="block")
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_players]
            mask = torch.repeat_interleave(mask, n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, n_features, dim=-1).expand(bs, -1, -1)

        if self.params["cuda"]:
            mask, time_gap = mask.to(device), time_gap.to(device)

        input_dict["mask"] = mask
        input_dict["input"] = input_data * mask  # masking missing values
        input_dict["delta"] = time_gap

        ret_f = self.rits_f(input_dict)
        ret_b = self.reverse(self.rits_b(self.reverse(input_dict)))
        ret = self.merge_ret(ret_f, ret_b, mode)

        if self.params["xy_sort"]:
            ret["pred"] = xy_sort_tensor(ret["pred"], sort_idxs, total_players, mode="restore")
            ret["target"] = xy_sort_tensor(ret["target"], sort_idxs, total_players, mode="restore")
            ret["mask"] = xy_sort_tensor(ret["mask"], sort_idxs, total_players, mode="restore")

        return ret

    def merge_ret(self, ret_f, ret_b, mode):
        loss_f = ret_f["loss"]
        loss_b = ret_b["loss"]
        loss_c = self.get_consistency_loss(ret_f["pred"], ret_b["pred"])

        loss = loss_f + loss_b + loss_c

        pred = (ret_f["pred"] + ret_b["pred"]) / 2
        target = ret_f["target"]
        mask = ret_f["mask"]

        aggfunc = "mean" if mode == "train" else "sum"
        pos_dist = calc_trace_dist(pred, target, mask, aggfunc=aggfunc, dataset=self.params["dataset"])

        ret_f["total_loss"] = loss
        ret_f["pred_dist"] = pos_dist
        ret_f["pred"] = pred

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            device = tensor_.device
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            indices = indices.to(device)

            return tensor_.index_select(1, indices)

        for key in ret:
            if not key.endswith("_loss"):
                ret[key] = reverse_tensor(ret[key])

        return ret
