import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from models.dbhp.dbhp_imputer import DBHPImputer
from models.dbhp.utils import calc_static_hybrid_pred, calc_static_hybrid_pred2
from models.utils import *


class DBHP(nn.Module):
    def __init__(self, params, parser=None):
        super(DBHP, self).__init__()

        self.model_args = [
            "missing_pattern",
            "team_size",
            "pe_z_dim",
            "pi_z_dim",
            "rnn_dim",
            "hybrid_rnn_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "physics_loss",
            "coherence_loss",
            "cartesian_accel",
            "transformer",
            "ppe",
            "fpe",
            "fpi",
            "train_hybrid",
            "bidirectional",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        self.model = DBHPImputer(self.params)

    def forward(self, data: Tuple[torch.Tensor], mode="train", device="cuda:0"):
        if mode == "test":
            if self.params["dataset"] == "afootball":  # randomly permute the player order
                player_data = random_permutation(data[0], 6)
            else:
                player_data, sort_idxs = xy_sort_tensor(data[0], n_players=self.params["team_size"] * 2)
        else:
            player_data = data[0]

        if self.params["dataset"] == "soccer":
            ball_data = data[1]
        else:
            ball_data = None

        data_dict = {"target": player_data, "ball": ball_data}
        bs, seq_len = data_dict["target"].shape[:2]

        missing_probs = np.arange(10) * 0.1
        mask = generate_mask(
            data_dict=data_dict,
            sports=self.params["dataset"],
            mode=self.params["missing_pattern"],
            window_size=seq_len,
            missing_rate=missing_probs[random.randint(1, 9)],
        )

        if self.params["missing_pattern"] == "camera":
            deltas_f, deltas_b = compute_deltas(mask)

            mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, agents]
            mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1)  # [bs, time, x]

            deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)  # [bs, time, agents]
            deltas_b = torch.tensor(deltas_b.copy(), dtype=torch.float32)

            if mode == "test":  # For section 5
                ball = data_dict["ball"].clone().cpu()
                if self.params["normalize"]:
                    ball_loc_unnormalized = normalize_tensor(ball, mode="upscale", dataset=self.params["dataset"])
                    poly_coords = compute_camera_coverage(ball_loc_unnormalized)
        else:  # if missing_pattern in ["uniform", "playerwise"]
            mask_ = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).expand(bs, -1, -1)  # [bs, time, agents]
            deltas_f, deltas_b = compute_deltas(np.array(mask_))

            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_agents]
            mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1).expand(bs, -1, -1)  # [bs, time, x]

            deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)  # [bs, time, n_agents]
            deltas_b = torch.tensor(deltas_b.copy(), dtype=torch.float32)

        if self.params["cuda"]:
            mask, deltas_f, deltas_b = mask.to(device), deltas_f.to(device), deltas_b.to(device)

        masked_input = player_data * mask

        data_dict["mask"] = mask
        data_dict["deltas_f"] = deltas_f
        data_dict["deltas_b"] = deltas_b
        data_dict["input"] = masked_input

        ret = self.model(data_dict, device=device)

        if mode == "test" and self.params["train_hybrid"]:
            ret["static_hybrid_pred"] = calc_static_hybrid_pred(ret)
            ret["static_hybrid2_pred"] = calc_static_hybrid_pred2(ret)

        aggfunc = "mean" if mode == "train" else "sum"
        pred_keys = ["pred"]
        if self.params["physics_loss"]:
            pred_keys += ["physics_f", "physics_b"]
        if self.params["train_hybrid"]:
            if mode == "test":
                pred_keys += ["static_hybrid", "static_hybrid2"]
            pred_keys += ["train_hybrid"]
        for key in pred_keys:
            if key == "pred":
                ret[f"{key}_dist"] = calc_trace_dist(
                    ret[f"{key}"],
                    ret["target"],
                    ret["mask"],
                    aggfunc=aggfunc,
                    dataset=self.params["dataset"],
                )
            else:
                ret[f"{key}_dist"] = calc_trace_dist(
                    ret[f"{key}_pred"],
                    ret["target"],
                    ret["mask"],
                    aggfunc=aggfunc,
                    dataset=self.params["dataset"],
                )

        if mode == "test" and self.params["missing_pattern"] == "camera":  # for section 5
            ret["polygon_points"] = poly_coords

        if mode == "test" and self.params["dataset"] != "afootball":
            for key in pred_keys:
                if key == "pred":
                    ret[key] = xy_sort_tensor(ret[key], sort_idxs, self.params["team_size"] * 2, mode="restore")
                else:
                    ret[f"{key}_pred"] = xy_sort_tensor(
                        ret[f"{key}_pred"], sort_idxs, self.params["team_size"] * 2, mode="restore"
                    )
            ret["target"] = xy_sort_tensor(ret["target"], sort_idxs, self.params["team_size"] * 2, mode="restore")
            ret["mask"] = xy_sort_tensor(ret["mask"], sort_idxs, self.params["team_size"] * 2, mode="restore")

        return ret
