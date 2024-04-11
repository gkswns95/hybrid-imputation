from typing import Dict

import torch

from models.utils import get_dataset_config


def ffill(t: torch.Tensor) -> torch.Tensor:  # [bs * agents, time, feats]
    idx0 = torch.tile(torch.arange(t.shape[0]), (t.shape[1], 1)).T.flatten()  # [bs * agents * time]
    idx1 = torch.arange(t.shape[1]).unsqueeze(0).expand(t.shape[0], t.shape[1]).clone()
    idx1[t[..., 0] == 0] = 0
    idx1 = (idx1.cummax(axis=1)[0]).flatten()  # [bs * agents * time]
    return t[tuple([idx0, idx1])].reshape(t.shape)  # [bs * agents, time, feats]


def deriv_accum_pred(data: Dict[str, torch.Tensor], use_accel=False, fb="f", dataset="soccer") -> torch.Tensor:
    n_players, ps = get_dataset_config(dataset)
    if dataset == "football":
        ps = (1, 1)

    scale_tensor = torch.FloatTensor(ps).to(data["pos_pred"].device)
    m = data["pos_mask"].flatten(0, 1)  # [bs * players, time, 2]
    dp_pos = data["pos_pred"].flatten(0, 1) * scale_tensor
    dp_vel = data["vel_pred"].flatten(0, 1)

    if use_accel:
        dp_accel = data["cartesian_accel_pred"].flatten(0, 1)

        if fb == "f":
            va = (dp_vel * 0.1 + dp_accel * 0.01).roll(1, dims=1)  # [bs * players, time, 2]
            cumsum_va = ((1 - m) * va).cumsum(axis=1)
            cumsum_va_by_segment = (1 - m) * (cumsum_va - ffill(m * cumsum_va))
            dap_pos = m * dp_pos + (1 - m) * (ffill(m * dp_pos) + cumsum_va_by_segment)

        else:
            dp_pos = torch.flip(dp_pos, dims=(1,))  # [bs * players, time, 2]
            dp_vel = torch.flip(dp_vel, dims=(1,)).roll(2, dims=1)
            dp_accel = torch.flip(dp_accel, dims=(1,)).roll(1, dims=1)
            dp_vel[:, 1] = dp_vel[:, 2].clone()
            m = torch.flip(m, dims=(1,))

            va = -dp_vel * 0.1 + dp_accel * 0.01
            cumsum_va = ((1 - m) * va).cumsum(axis=1)
            cumsum_va_by_segment = (1 - m) * (cumsum_va - ffill(m * cumsum_va))
            dap_pos = torch.flip(m * dp_pos + (1 - m) * (ffill(m * dp_pos) + cumsum_va_by_segment), dims=(1,))

    else:
        if fb == "f":
            cumsum_v = ((1 - m) * dp_vel * 0.1).cumsum(axis=1)  # [bs * players, time, 2]
            cumsum_v_by_segment = (1 - m) * (cumsum_v - ffill(m * cumsum_v))
            dap_pos = m * dp_pos + (1 - m) * (ffill(m * dp_pos) + cumsum_v_by_segment)

        else:
            dp_pos = torch.flip(dp_pos, dims=(1,))  # [bs * players, time, 2]
            dp_vel = torch.flip(dp_vel, dims=(1,)).roll(1, dims=1)
            m = torch.flip(m, dims=(1,))

            cumsum_v = ((1 - m) * -dp_vel * 0.1).cumsum(axis=1)
            cumsum_v_by_segment = (1 - m) * (cumsum_v - ffill(m * cumsum_v))
            dap_pos = torch.flip(m * dp_pos + (1 - m) * (ffill(m * dp_pos) + cumsum_v_by_segment), dims=(1,))

    return dap_pos.reshape(-1, n_players, dap_pos.shape[1], 2) / scale_tensor  # [bs, players, time, 2]


def static_hybrid_pred2(ret):  # Mixing DAP-F and DAP-B
    deltas_f, deltas_b = ret["deltas_f"], ret["deltas_b"]  # [bs, time, players]
    deltas_f = deltas_f.transpose(1, 2).flatten(0, 1).unsqueeze(-1)  # [bs * agents, time, 1]
    deltas_b = deltas_b.transpose(1, 2).flatten(0, 1).unsqueeze(-1)

    masks = ret["pos_mask"].flatten(0, 1)[..., 0:1]  # [bs * players, time, 1]

    t = torch.arange(masks.shape[1]).reshape(1, -1, 1).tile((masks.shape[0], 1, 1)).to(masks.device)
    t0 = t - deltas_f
    t1 = t + deltas_b
    m = (t0 + t1) / 2

    wf = torch.nan_to_num((t1 - t) / (t1 - t0), 0)
    wb = torch.nan_to_num((t - t0) / (t1 - t0), 0)

    # print(torch.cat([t, masks_, t0, m, t1, wf, (1 - masks_) * wb], dim=-1)[1, :20])

    bs, seq_len, _ = ret["dap_f"].shape
    dap_f = ret["dap_f"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)
    dap_b = ret["dap_b"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)

    hybrid_pos = wf * dap_f + wb * dap_b  # [bs * agents, time, 2]
    hybrid_pos = hybrid_pos.reshape(bs, -1, seq_len, 2).transpose(1, 2).flatten(2, 3)

    return hybrid_pos


def static_hybrid_pred(ret):  # Mixing DP, DAP-F, and DAP-B
    deltas_f, deltas_b = ret["deltas_f"], ret["deltas_b"]  # [bs, time, agents]
    deltas_f = deltas_f.transpose(1, 2).flatten(0, 1).unsqueeze(-1)  # [bs * agents, time, 1]
    deltas_b = deltas_b.transpose(1, 2).flatten(0, 1).unsqueeze(-1)

    masks = ret["pos_mask"].flatten(0, 1)[..., 0:1]  # [bs * players, time, 1]

    t = torch.arange(masks.shape[1]).reshape(1, -1, 1).tile((masks.shape[0], 1, 1)).to(masks.device)
    t0 = t - deltas_f
    t1 = t + deltas_b
    m = (t0 + t1) / 2

    w0 = (t - t0) / (m - t0)
    w1 = (t1 - t) / (t1 - m)
    is_front = (t < m).float()

    wd = torch.nan_to_num(is_front * w0 + (1 - is_front) * w1, 0)  # [bs * agents, time, 1]
    wf = is_front * (1 - wd)
    wb = (1 - is_front) * (1 - wd)

    # print(torch.cat([t, masks_, wd, wf, (1 - masks_) * wb], dim=-1)[1, :20])

    bs, seq_len, _ = ret["dap_f"].shape
    dp_pos = ret["pos_pred"].flatten(0, 1)  # [bs * agents, time, 2]
    dap_f = ret["dap_f"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)
    dap_b = ret["dap_b"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)

    hybrid_pos = wd * dp_pos + wf * dap_f + wb * dap_b  # [bs * agents, time, 2]
    hybrid_pos = hybrid_pos.reshape(bs, -1, seq_len, 2).transpose(1, 2).flatten(2, 3)

    return hybrid_pos
