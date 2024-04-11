import torch

from models.utils import get_dataset_config


def ffill(t: torch.Tensor) -> torch.Tensor:  # [bs * agents, time, feats]
    idx0 = torch.tile(torch.arange(t.shape[0]), (t.shape[1], 1)).T.flatten()  # [bs * agents * time]
    idx1 = torch.arange(t.shape[1]).unsqueeze(0).expand(t.shape[0], t.shape[1]).clone()
    idx1[t[..., 0] == 0] = 0
    idx1 = (idx1.cummax(axis=1)[0]).flatten()  # [bs * agents * time]
    return t[tuple([idx0, idx1])].reshape(t.shape)  # [bs * agents, time, feats]


def derivative_based_pred(ret, physics_mode="vel", fb="f", dataset="soccer"):
    n_agents, ps = get_dataset_config(dataset)
    if dataset == "football":
        ps = (1, 1)

    scale_tensor = torch.FloatTensor(ps).to(ret["pos_pred"].device)
    m = ret["pos_mask"].flatten(0, 1)  # [bs * n_agents, time, 2]
    pxy_d = ret["pos_pred"].flatten(0, 1) * scale_tensor
    vxy_d = ret["vel_pred"].flatten(0, 1)

    if physics_mode == "vel":
        if fb == "f":
            cumsum_v = ((1 - m) * vxy_d * 0.1).cumsum(axis=1)  # [bs * n_agents, time, 2]
            cumsum_v_by_segment = (1 - m) * (cumsum_v - ffill(m * cumsum_v))
            pxy_fb = m * pxy_d + (1 - m) * (ffill(m * pxy_d) + cumsum_v_by_segment)

        else:
            pxy_d = torch.flip(pxy_d, dims=(1,))  # [bs * n_agents, time, 2]
            vxy_d = torch.flip(vxy_d, dims=(1,)).roll(1, dims=1)
            m = torch.flip(m, dims=(1,))

            cumsum_v = ((1 - m) * -vxy_d * 0.1).cumsum(axis=1)
            cumsum_v_by_segment = (1 - m) * (cumsum_v - ffill(m * cumsum_v))
            pxy_fb = torch.flip(m * pxy_d + (1 - m) * (ffill(m * pxy_d) + cumsum_v_by_segment), dims=(1,))
    else:
        axy_d = ret["cartesian_accel_pred"].flatten(0, 1)

        if fb == "f":
            va = (vxy_d * 0.1 + axy_d * 0.01).roll(1, dims=1)  # [bs * n_agents, time, 2]
            cumsum_va = ((1 - m) * va).cumsum(axis=1)
            cumsum_va_by_segment = (1 - m) * (cumsum_va - ffill(m * cumsum_va))
            pxy_fb = m * pxy_d + (1 - m) * (ffill(m * pxy_d) + cumsum_va_by_segment)

        else:
            pxy_d = torch.flip(pxy_d, dims=(1,))  # [bs * n_agents, time, 2]
            vxy_d = torch.flip(vxy_d, dims=(1,)).roll(2, dims=1)
            axy_d = torch.flip(axy_d, dims=(1,)).roll(1, dims=1)
            vxy_d[:, 1] = vxy_d[:, 2].clone()
            m = torch.flip(m, dims=(1,))

            va = -vxy_d * 0.1 + axy_d * 0.01
            cumsum_va = ((1 - m) * va).cumsum(axis=1)
            cumsum_va_by_segment = (1 - m) * (cumsum_va - ffill(m * cumsum_va))
            pxy_fb = torch.flip(m * pxy_d + (1 - m) * (ffill(m * pxy_d) + cumsum_va_by_segment), dims=(1,))

    return pxy_fb.reshape(-1, n_agents, pxy_fb.shape[1], 2) / scale_tensor  # [bs, n_agents, time, 2]


def calc_static_hybrid_pred2(ret):  # Mixing DAP-F and DAP-B
    """
    2 preds
    """
    deltas_f, deltas_b = ret["deltas_f"], ret["deltas_b"]  # [bs, time, n_agents]
    deltas_f = deltas_f.transpose(1, 2).flatten(0, 1).unsqueeze(-1)  # [bs * agents, time, 1]
    deltas_b = deltas_b.transpose(1, 2).flatten(0, 1).unsqueeze(-1)

    masks = ret["pos_mask"].flatten(0, 1)[..., 0:1]  # [bs * n_agents, time, 1]

    t = torch.arange(masks.shape[1]).reshape(1, -1, 1).tile((masks.shape[0], 1, 1)).to(masks.device)
    t0 = t - deltas_f
    t1 = t + deltas_b
    m = (t0 + t1) / 2

    wf = torch.nan_to_num((t1 - t) / (t1 - t0), 0)
    wb = torch.nan_to_num((t - t0) / (t1 - t0), 0)

    # print(torch.cat([t, masks_, t0, m, t1, wf, (1 - masks_) * wb], dim=-1)[1, :20])

    bs, seq_len, _ = ret["physics_f_pred"].shape
    xy_pred = ret["pos_pred"].transpose(1, 2).flatten(0, 1)  # [bs * agents, time, 2]
    pred_f = ret["physics_f_pred"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)
    pred_b = ret["physics_b_pred"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)

    static_hybrid_pred = wf * pred_f + wb * pred_b  # [bs * agents, time, 2]
    static_hybrid_pred = static_hybrid_pred.reshape(bs, -1, seq_len, 2).transpose(1, 2).flatten(2, 3)

    return static_hybrid_pred


def calc_static_hybrid_pred(ret):  # Mixing DP, DAP-F, and DAP-B
    deltas_f, deltas_b = ret["deltas_f"], ret["deltas_b"]  # [bs, time, agents]
    deltas_f = deltas_f.transpose(1, 2).flatten(0, 1).unsqueeze(-1)  # [bs * agents, time, 1]
    deltas_b = deltas_b.transpose(1, 2).flatten(0, 1).unsqueeze(-1)

    masks = ret["pos_mask"].flatten(0, 1)[..., 0:1]  # [bs * n_agents, time, 1]

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

    bs, seq_len, _ = ret["physics_f_pred"].shape
    xy_pred = ret["pos_pred"].flatten(0, 1)  # [bs * agents, time, 2]
    pred_f = ret["physics_f_pred"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)
    pred_b = ret["physics_b_pred"].reshape(bs, seq_len, -1, 2).transpose(1, 2).flatten(0, 1)

    static_hybrid_pred = wd * xy_pred + wf * pred_f + wb * pred_b  # [bs * agents, time, 2]
    static_hybrid_pred = static_hybrid_pred.reshape(bs, -1, seq_len, 2).transpose(1, 2).flatten(2, 3)

    return static_hybrid_pred
