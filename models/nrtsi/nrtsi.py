import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.nrtsi.nrtsi_imputer import NRTSIImputer
from models.nrtsi.utils import (
    gap_to_max_gap,
    get_next_to_impute,
    nll_gauss,
    sample_gauss,
)
from models.utils import *


class NRTSI(nn.Module):
    def __init__(self, params, parser=None):
        super(NRTSI, self).__init__()
        self.model_args = [
            "missing_pattern",
            "n_players",
            "n_max_time_scale",
            "time_enc_dim",
            "att_dim",
            "model_dim",
            "inner_dim",
            "time_dim",
            "expand_dim",
            "n_layers",
            "n_heads",
            "n_max_level",
            "cartesian_accel",
            "use_mask",
            "xy_sort",
            "stochastic",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        self.model = NRTSIImputer(self.params)

    def forward(self, data, model, gap_models=None, mode="train", teacher_forcing=False, device="cuda:0"):
        ret = {"loss": 0, "pred_dist": 0}

        n_features = self.params["n_features"]
        dataset = self.params["dataset"]

        if dataset == "soccer":
            total_players = 22
        elif dataset == "basketball":
            total_players = 10
        else:  # e.g. "football"
            total_players = 6

        if self.params["xy_sort"]:
            input_data, sort_idxs = xy_sort_tensor(data[0], n_players=total_players)  # [bs, time, x_dim]
            target_data = input_data.clone()
        else:
            if dataset == "football":  # randomly permute player order for NFL dataset.
                data[0] = random_permutation(data[0], total_players)
                data[1] = data[0].clone()
            input_data = data[0]  # [bs, time, x_dim]
            target_data = data[1]

        if mode == "train":
            min_gap, max_gap = data[2], data[3]

        bs, seq_len = input_data.shape[0], input_data.shape[1]

        missing_probs = np.arange(10) * 0.1
        mask_data = generate_mask(
            inputs=ret,
            mode=self.params["missing_pattern"],
            ws=seq_len,
            missing_rate=missing_probs[random.randint(1, 9)],
            dataset=dataset,
        )
        mask_data = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
        mask_data = torch.repeat_interleave(mask_data, n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]

        if self.params["cuda"]:
            input_data, target_data, mask_data = input_data.to(device), target_data.to(device), mask_data.to(device)

        mask = mask_data.clone()

        total_loss = Variable(torch.tensor(0.0), requires_grad=True).to(device)

        num_levels = torch.tensor(1e-6)
        pos_dist, n_missings = torch.tensor(1e-6), torch.tensor(1e-6)

        if mode == "train":
            init_obs = True
            obs_list_count = (mask_data[0, :, 0] == 1).nonzero().reshape(-1).tolist()
            while len(obs_list_count) < seq_len:
                next_list_count, gap = get_next_to_impute(mask_data, self.params["n_max_level"])
                obs_list = torch.from_numpy(np.array(obs_list_count)).long()
                if self.params["cuda"]:
                    obs_list = obs_list.to(device)
                obs_list_count += next_list_count
                mask_data[:, next_list_count, :] = 1
                if min_gap < gap and gap <= max_gap:
                    if teacher_forcing or init_obs:
                        obs_data = input_data[:, obs_list, :]  # [bs, n_obs, x_dim]
                        init_obs = False
                    gt_data = target_data[:, next_list_count, :]  # [bs, n_imp, x_dim]

                    obs_list = obs_list[None, :, None].expand(bs, -1, -1)  # [bs, obs_len, 1]
                    next_list = torch.from_numpy(np.array(next_list_count)).long()[None, :, None]  # [1, imp_len, 1]
                    next_list = next_list.expand(bs, -1, -1)  # [bs, imp_len, 1]
                    if self.params["cuda"]:
                        next_list = next_list.to(device)
                    imputations = self.model(obs_data, obs_list, next_list, gap)  # [bs, n_imp, y_dim]
                    if self.params["stochastic"]:
                        gt_data_ = reshape_tensor(
                            gt_data, rescale=False, n_features=n_features, dataset=dataset
                        ).flatten(2, 3)
                        imputations_ = reshape_tensor(
                            imputations, rescale=False, n_features=n_features, dataset=dataset
                        ).flatten(2, 3)

                        total_loss += nll_gauss(gt_data_, imputations_)
                        imputations = sample_gauss(imputations, gt_data, gap=gap)
                    else:
                        total_loss += torch.mean(torch.abs(imputations - gt_data))

                    imputations_ = (
                        reshape_tensor(imputations, rescale=True, n_features=n_features, dataset=dataset).detach().cpu()
                    )  # [bs, n_imp, total_players, 2]
                    gt_data_ = (
                        reshape_tensor(gt_data, rescale=True, n_features=n_features, dataset=dataset).detach().cpu()
                    )
                    mask_ = (
                        reshape_tensor(mask, rescale=False, n_features=n_features, dataset=dataset).detach().cpu()
                    )  # [bs, time, total_players, 2]
                    pos_dist += torch.sum(torch.norm(imputations_ - gt_data_, dim=-1))
                    n_missings += (1 - mask_[:, next_list_count]).sum() / 2

                    num_levels += 1

                    if not teacher_forcing:
                        obs_data = torch.cat([obs_data, imputations], dim=1)  # [bs, n_imp + n_obs, y_dim]
            ret["total_loss"] = total_loss / num_levels
            ret["pred_dist"] = pos_dist / n_missings

        else:  # e.g. "test"
            n_samples = 1 if dataset == "football" else 1
            for _ in range(n_samples):
                init_obs = True
                pred = target_data.clone()
                obs_list_count = (mask_data[0, :, 0] == 1).nonzero().reshape(-1).tolist()
                while len(obs_list_count) < seq_len:
                    next_list_count, gap = get_next_to_impute(mask_data, self.params["n_max_level"])
                    if self.params["stochastic"] and gap > 2**2:  # section 3.3 in NRTSI paper (Stochastic Time Series)
                        next_list_count = [next_list_count[0]]

                    max_gap = gap_to_max_gap(gap)  # load best model.
                    assert gap_models is not None
                    model.load_state_dict(gap_models[max_gap])

                    obs_list = torch.from_numpy(np.array(obs_list_count)).long()
                    if self.params["cuda"]:
                        obs_list = obs_list.to(device)
                    obs_list_count += next_list_count
                    mask_data[:, next_list_count, :] = 1

                    if init_obs:
                        obs_data = input_data[:, obs_list, :]  # [bs, n_obs, feat_dim]
                        init_obs = False
                    gt_data = target_data[:, next_list_count, :]  # [bs, n_imp, feat_dim]

                    obs_list = obs_list[None, :, None].expand(bs, -1, -1)  # [bs, obs_len, 1]
                    next_list = torch.from_numpy(np.array(next_list_count)).long()[None, :, None]  # [1, imp_len, 1]
                    next_list = next_list.expand(bs, -1, -1)  # [bs, imp_len, 1]
                    if self.params["cuda"]:
                        next_list = next_list.to(device)

                    imputations = self.model(obs_data, obs_list, next_list, gap)  # [bs, n_imp, y_dim]

                    if self.params["stochastic"]:
                        # imputations = sample_gauss(pred_mean_, pred_std_, gt_data_, gap=gap)
                        # imputations = sample_gauss(imputations, gt_data, gap=gap)
                        # gt_data = reshape_tensor(
                        # gt_data, rescale=False, n_features=n_features, dataset=dataset).flatten(2,3)
                        imputations = sample_gauss(imputations, gt_data, gap=gap)

                    pred[:, next_list_count, :] = imputations

                    obs_data = torch.cat([obs_data, imputations], dim=1)  # [bs, n_imp + n_obs, y_dim]

                pred_ = reshape_tensor(pred, rescale=True, n_features=n_features, dataset=dataset)
                target_ = reshape_tensor(target_data, rescale=True, n_features=n_features, dataset=dataset)

                ret["pred_dist"] = torch.norm(pred_ - target_, dim=-1).sum()
                ret["pred"] = pred
                ret["target"] = target_data
                ret["mask"] = mask

                if self.params["xy_sort"]:
                    ret["pred"] = xy_sort_tensor(ret["pred"], sort_idxs, total_players, mode="restore")
                    ret["target"] = xy_sort_tensor(ret["target"], sort_idxs, total_players, mode="restore")
                    ret["mask"] = xy_sort_tensor(ret["mask"], sort_idxs, total_players, mode="restore")

        return ret

    def forward2(self, obs_data, obs_list, next_list, gap):
        prediction = self.model(obs_data, obs_list, next_list, gap).detach()

        return prediction
