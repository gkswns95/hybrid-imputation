import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from models.dbhp.dbhp_imputer import DBHPImputer
from models.dbhp.utils import static_hybrid_pred, static_hybrid_pred2
from models.utils import *


class DBHP(nn.Module):
    def __init__(self, params, parser=None):
        super(DBHP, self).__init__()

        self.model_args = [
            "team_size",
            "ppe",
            "fpe",
            "fpi",
            "transformer",
            "bidirectional",
            "pe_z_dim",
            "pi_z_dim",
            "rnn_dim",
            "hybrid_rnn_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "cartesian_accel",
            "deriv_accum",
            "dynamic_hybrid",
            "coherence_loss",
        ]
        self.params = parse_model_params(self.model_args, params, parser)

        print_args = ["ppe", "fpe", "fpi", "rnn_dim", "hybrid_rnn_dim", "deriv_accum", "dynamic_hybrid"]
        self.params_str = get_params_str(print_args, params)

        self.build()

    def build(self):
        self.model = DBHPImputer(self.params)

    def forward(self, data: Tuple[torch.Tensor], mode="train", device="cuda:0"):
        if mode == "test":
            if self.params["dataset"] == "afootball":  # randomly permute the player order
                player_data = random_permutation(data[0], 6)
            else:
                player_data, sort_idxs = xy_sort_tensor(data[0], players=self.params["team_size"] * 2)
        else:
            player_data = data[0]  # [bs, time, x] = [bs, time, players * feats]

        if self.params["dataset"] == "soccer":
            ball_data = data[1]  # [bs, time, 2]
        else:
            ball_data = None

        ret = {"target": player_data, "ball": ball_data}

        mask = generate_mask(
            data=ret,
            sports=self.params["dataset"],
            mode=self.params["missing_pattern"],
            missing_rate=random.randint(1, 9) * 0.1,
        )  # [bs, time, players]
        deltas_f, deltas_b = compute_deltas(mask)  # [bs, time, players]

        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, players]
        mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1)  # [bs, time, x]
        deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)
        deltas_b = torch.tensor(deltas_b.copy(), dtype=torch.float32)

        if self.params["missing_pattern"] == "camera" and mode == "test":  # for section 5
            ball_data = ret["ball"].clone().cpu()
            if self.params["normalize"]:
                ball_loc_unnormalized = normalize_tensor(ball_data, mode="upscale", dataset=self.params["dataset"])
                camera_vertices = compute_camera_coverage(ball_loc_unnormalized)

        # else:  # if missing_pattern in ["uniform", "playerwise"]
        #     mask_ = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).expand(bs, -1, -1)  # [bs, time, players]
        #     deltas_f, deltas_b = compute_deltas(np.array(mask_))

        #     mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, players]
        #     mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1).expand(bs, -1, -1)
        #     deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)  # [bs, time, players]
        #     deltas_b = torch.tensor(deltas_b.copy(), dtype=torch.float32)

        if self.params["cuda"]:
            mask, deltas_f, deltas_b = mask.to(device), deltas_f.to(device), deltas_b.to(device)

        ret["mask"] = mask
        ret["input"] = player_data * mask
        ret["deltas_f"] = deltas_f
        ret["deltas_b"] = deltas_b

        ret = self.model(ret, device=device)

        if mode == "test" and self.params["dynamic_hybrid"]:
            ret["hybrid_s"] = static_hybrid_pred(ret)
            ret["hybrid_s2"] = static_hybrid_pred2(ret)

        aggfunc = "mean" if mode == "train" else "sum"
        pred_keys = ["pred"]
        if self.params["deriv_accum"]:
            pred_keys += ["dap_f", "dap_b"]
        if self.params["dynamic_hybrid"]:
            if mode == "test":
                pred_keys += ["hybrid_s", "hybrid_s2"]
            pred_keys += ["hybrid_d"]

        for k in pred_keys:
            ret[f"{k}_dist"] = calc_trace_dist(
                ret[k],
                ret["target"],
                ret["mask"],
                aggfunc=aggfunc,
                dataset=self.params["dataset"],
            )

        if mode == "test" and self.params["missing_pattern"] == "camera":  # for section 5
            ret["camera_vertices"] = camera_vertices

        if mode == "test" and self.params["dataset"] != "afootball":
            for k in pred_keys:
                if k == "pred":
                    ret[k] = xy_sort_tensor(ret[k], sort_idxs, self.params["team_size"] * 2, mode="restore")
                else:
                    ret[f"{k}_pred"] = xy_sort_tensor(
                        ret[f"{k}_pred"], sort_idxs, self.params["team_size"] * 2, mode="restore"
                    )
            ret["target"] = xy_sort_tensor(ret["target"], sort_idxs, self.params["team_size"] * 2, mode="restore")
            ret["mask"] = xy_sort_tensor(ret["mask"], sort_idxs, self.params["team_size"] * 2, mode="restore")

        return ret
