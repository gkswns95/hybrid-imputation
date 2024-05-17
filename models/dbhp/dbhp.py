import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from models.dbhp.dbhp_forecaster import DBHPForecaster
from models.dbhp.dbhp_imputer import DBHPImputer
from models.dbhp.utils import static_hybrid_pred, static_hybrid_pred2
from models.utils import *


class DBHP(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "team_size",
            "uniagent",
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

        print_args = ["uniagent", "ppe", "fpe", "fpi", "deriv_accum", "dynamic_hybrid"]
        self.params_str = get_params_str(print_args, params)

        self.build()

    def build(self):
        if self.params["missing_pattern"] == "forecast":
            self.model = DBHPForecaster(self.params)
        else:
            self.model = DBHPImputer(self.params)

    def forward(self, data: Tuple[torch.Tensor], mode="train", device="cuda:0"):
        if "uniagent" not in self.params:
            self.params["uniagent"] = False
        if "player_order" not in self.params:
            self.params["player_order"] = None

        # if mode == "test" and self.params["dataset"] == "afootball":  # shuffle the players' order
        if self.params["player_order"] == "shuffle":
            player_data, player_orders = shuffle_players(data[0], n_players=self.params["team_size"] * 2)
        elif mode == "test" or self.params["player_order"] == "xy_sort":  # sort players by x+y values
            player_data, player_orders = sort_players(data[0], n_players=self.params["team_size"] * 2)
        else:
            player_data, player_orders = data[0], None  # [bs, time, x] = [bs, time, players * feats]

        ball_data = data[1] if self.params["dataset"] == "soccer" else None  # [bs, time, 2]
        ret = {"target": player_data, "ball": ball_data}

        mask, missing_rate = generate_mask(
            data=ret,
            sports=self.params["dataset"],
            mode=self.params["missing_pattern"],
            missing_rate=self.params["missing_rate"],
        )  # [bs, time, players]

        if self.params["missing_pattern"] == "forecast":
            deltas_f = compute_deltas(mask, bidirectional=False)  # [bs, time, players]
            deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)
            if self.params["cuda"]:
                deltas_f = deltas_f.to(device)

            ret["deltas_f"] = deltas_f

        else:
            deltas_f, deltas_b = compute_deltas(mask)  # [bs, time, players]
            deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)
            deltas_b = torch.tensor(deltas_b.copy(), dtype=torch.float32)
            if self.params["cuda"]:
                deltas_f, deltas_b = deltas_f.to(device), deltas_b.to(device)

            ret["deltas_f"] = deltas_f
            ret["deltas_b"] = deltas_b

        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, players]
        mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1)  # [bs, time, x]
        if self.params["cuda"]:
            mask = mask.to(device)

        ret["mask"] = mask
        ret["input"] = player_data * mask
        ret["missing_rate"] = missing_rate

        if mode == "test" and self.params["missing_pattern"] == "camera":  # for section 5
            ball_data = ret["ball"].clone().cpu()
            if self.params["normalize"]:
                ball_rescaled = normalize_tensor(ball_data, mode="upscale", dataset_type=self.params["dataset"])
                ret["camera_vertices"] = compute_camera_coverage(ball_rescaled)
            else:
                ret["camera_vertices"] = compute_camera_coverage(ball_data)

        # else:  # if missing_pattern in ["uniform", "playerwise"]
        #     mask_ = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).expand(bs, -1, -1)  # [bs, time, players]
        #     deltas_f, deltas_b = compute_deltas(np.array(mask_))

        #     mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, players]
        #     mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1).expand(bs, -1, -1)
        #     deltas_f = torch.tensor(deltas_f.copy(), dtype=torch.float32)  # [bs, time, players]
        #     deltas_b = torch.tensor(deltas_b.copy(), dtype=torch.float32)

        ret = self.model(ret, device=device)

        if mode == "test" and self.params["dynamic_hybrid"] and self.params["missing_pattern"] != "forecast":
            ret["hybrid_s"] = static_hybrid_pred(ret)
            ret["hybrid_s2"] = static_hybrid_pred2(ret)

        aggfunc = "mean" if mode == "train" else "sum"
        pred_keys = ["pred"]
        if self.params["deriv_accum"]:
            pred_keys += ["dap_f"]
            if self.params["missing_pattern"] != "forecast":
                pred_keys += ["dap_b"]
        if self.params["dynamic_hybrid"]:
            if mode == "test" and self.params["missing_pattern"] != "forecast":
                pred_keys += ["hybrid_s", "hybrid_s2"]
            pred_keys += ["hybrid_d"]

        for k in pred_keys:
            ret[f"{k}_pe"] = calc_pos_error(
                ret[k],
                ret["target"],
                ret["mask"],
                aggfunc=aggfunc,
                dataset=self.params["dataset"],
            )

        if player_orders is not None:
            ret["input"] = sort_players(ret["input"], player_orders, self.params["team_size"] * 2, mode="restore")
            ret["mask"] = sort_players(ret["mask"], player_orders, self.params["team_size"] * 2, mode="restore")
            ret["target"] = sort_players(ret["target"], player_orders, self.params["team_size"] * 2, mode="restore")
            for k in pred_keys:
                ret[k] = sort_players(ret[k], player_orders, self.params["team_size"] * 2, mode="restore")

        return ret
