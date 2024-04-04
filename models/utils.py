import time as t

import numpy as np
import torch
import torch.nn.functional as F
import json
import models
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from copy import deepcopy
from shapely.geometry import Polygon, Point
from torch.autograd import Variable
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import KNNImputer

def get_dataset_config(dataset):
    '''
    players : number of total players contained each dataset
    ps : (width, height) pitch sizes
    '''
    if dataset == "soccer":
        players = 22
        ps = (108, 72)
    elif dataset == "basketball":
        players = 10
        ps = (28.65, 15.24)
    elif dataset == "football":
        players = 6
        ps = (110, 49)
        # ps = (1, 1)
    
    return players, ps

def is_inside(polygon, point):
    def cross(p1, p2, on_line_mask, counters):        
        x1, y1 = p1[0], p1[1] # [bs, time]
        x2, y2 = p2[0], p2[1]
        players_x, players_y = point[..., 0], point[..., 1] # [bs, time, n_players]

        for p in range(n_players):
            x_p = players_x[..., p] # [bs, time]
            y_p = players_y[..., p] 
            
            # horizontal line
            condition1 = (y1 - y2 == 0) # [bs, time]
            condition2 = (y1 == y_p)
            condition3 = (torch.min(x1, x2) <= x_p) & (x_p <= torch.max(x1, x2))
            
            on_line_indices = (condition1 & condition2 & condition3).nonzero() # [bs, time]
            if len(on_line_indices):            
                on_line_mask[on_line_indices[:,0], on_line_indices[:,1], p] = 1

            # vertical line
            condition4 = (x1 - x2 == 0)
            condition5 = (torch.min(y1, y2) <= y_p) & (y_p <= torch.max(y1, y2))
            condition6 = (x_p <= torch.max(x1, x2))
            condition7 = (x_p == torch.max(x1, x2))

            counter_indices = (condition4 & condition5 & condition6).nonzero() # [bs, time]
            if len(counter_indices):
                counters[counter_indices[:,0], counter_indices[:,1], p] += 1

            on_line_indices = (condition4 & condition5 & condition6 & condition7).nonzero() # [bs, time]
            if len(on_line_indices):
                on_line_mask[on_line_indices[:,0], on_line_indices[:,1], p ] = 1

            # diagonal line
            a = (y1 - y2) / (x1 - x2)
            b = y1 - x1 * a
            x = (y_p - b) / a
        
            condition8 = (x_p <= x)
            condition9 = (torch.min(y1, y2) <= y_p) & (y_p <= torch.max(y1, y2))
            condition10 = (x_p == x) | (y_p == y1) | (y_p == y2)

            counter_indices = (condition8 & condition9).nonzero() # [bs, time]
            if len(counter_indices):
                counters[counter_indices[:,0], counter_indices[:,1], p] += 1

            on_line_indices = (condition8 & condition9 & condition10).nonzero() # [bs, time]
            if len(on_line_indices):
                on_line_mask[counter_indices[:,0], counter_indices[:,1], p] += 1

        return on_line_mask, counters

    # MAIN
    bs, seq_len, n_players = point.shape[:3]
    on_line_mask = torch.zeros(bs, seq_len, n_players)  
    counters = torch.zeros(bs, seq_len, n_players)

    for x in range(len(polygon)):
        on_line_mask, counters = cross(polygon[x], polygon[x-1], on_line_mask, counters)

    is_contain_mask = (counters % 2 == 0)

    missing_mask = is_contain_mask + on_line_mask

    missing_mask_np = np.array((1-missing_mask))
    return missing_mask_np

def generate_mask(inputs, mode="random", ws=100, missing_rate=0.8, dataset="soccer"):
    
    assert mode in ["player_wise", "all_player", "camera_simulate"]

    total_players, _ = get_dataset_config(dataset)
    n_missing = int(ws * total_players * missing_rate) # total missing values
    min_len = int(ws * 0.3)
    max_len = int(ws * 0.9)
    if mode == "player_wise":
        player_list = list(range(total_players))
        episode_mask = np.ones((ws, total_players))
        while n_missing > 0:
            missing_player = random.choice(player_list)
            if n_missing < min_len:
                block_len = random.randint(1, n_missing)
            else:
                block_len = random.randint(min_len, max_len)

            start_idx = random.randint(1, ws-block_len)
            end_idx = start_idx + block_len

            overlap_len = (episode_mask[start_idx: end_idx, missing_player] == 0).sum()
            episode_mask[start_idx: end_idx, missing_player] = 0

            block_len -= overlap_len
            n_missing -= block_len

            if n_missing <= 0:
                break
        
        episode_mask[0] = 1
        episode_mask[-1] = 1

    elif mode == "all_player":
        if dataset == "football":
            episode_mask = np.ones((ws, total_players))
            num_missing = random.randint(40, 49)
            missing_list_np = np.array(random.sample(range(ws), num_missing))
            episode_mask[missing_list_np] = 0
            episode_mask[0] = 1
            episode_mask[-1] = 1
        else:
            block_len = int(ws * missing_rate) # total missing values
            episode_mask = np.ones((ws, total_players))

            start_idx = random.randint(1, ws-block_len)
            end_idx = start_idx + block_len
            
            episode_mask[start_idx: end_idx, :] = 0
        
            episode_mask[0] = 1
            episode_mask[-1] = 1

    elif mode == "camera_simulate":
        target, ball = inputs["target"].clone().cpu(), inputs["ball"].clone().cpu()
        target_xy = reshape_tensor(target, rescale=True, dataset=dataset) # [bs, time, players, 2]
        ball_loc = normalize_tensor(ball, mode="reverse", dataset=dataset)

        bs = target.shape[0]
        poly_coords = compute_polygon_loc(ball_loc)

        # Check whether the camera view contains the player's (x, y) position or not
        # is_contains => On-screen players : 1, Off-screen players : 0
        episode_mask = is_inside(poly_coords, target_xy) # [bs, time, players]
        episode_mask[:, :5, :] = 1
        episode_mask[:, -5:, :] = 1

    return episode_mask

def compute_deltas(masks: np.ndarray):
    '''
    masks : [bs, time, n_agents]
    '''
    cumsum_rmasks_f = (1 - masks).cumsum(axis=1)
    cumsum_prevs_f = np.maximum.accumulate(cumsum_rmasks_f * masks, axis=1)
    deltas_f = cumsum_rmasks_f - cumsum_prevs_f

    cumsum_rmasks_b = np.flip(1 - masks, axis=1).cumsum(axis=1)
    cumsum_prevs_b = np.maximum.accumulate(cumsum_rmasks_b * np.flip(masks, axis=1), axis=1)
    deltas_b = np.flip(cumsum_rmasks_b - cumsum_prevs_b, axis=1)

    return deltas_f, deltas_b

def time_interval(masks, time_gap, direction="f", mode="block"):
    '''
    masks : [bs, time, n_players]
    '''
    if direction=="b":
        masks = np.flip(deepcopy(masks), axis=[0])

    deltas = np.zeros(masks.shape)
    if mode == "block":
        for t in range(1, masks.shape[0]):
            gap = time_gap[t] - time_gap[t-1]
            for p, m in enumerate(masks[t-1]):
                deltas[t, p] = gap + deltas[t-1, p] if m == 0 else gap
    elif mode == "camera":
        for batch in range(deltas.shape[0]):
            masks_ = masks[batch] # [time, n_players]
            for t in range(1, masks.shape[1]):
                gap = time_gap[t] - time_gap[t-1]  
                for p, m in enumerate(masks_[t-1]):
                    deltas[batch, t, p] = gap + deltas[batch, t-1, p] if m == 0 else gap

    return torch.tensor(deltas, dtype=torch.float32)

def random_permutation(input, players=6, permutations=None):
    bs, seq_len = input.shape[:2]
    input_ = input.reshape(bs, seq_len, players, -1) # [bs, time, players, feat_dim]
    
    permuted_input = input_.clone()
    permutations = np.zeros((bs, players))
    for batch in range(bs):
        # rand_idx = np.random.permutation(players)  
        rand_idx = [0,1,2,3,4,5]  
        random.shuffle(rand_idx)
        permuted_input[batch, :] = input_[batch, :, rand_idx]
        permutations[batch] = rand_idx

    return permuted_input.flatten(2,3)

# def xy_sort_tensor(tensor, n_players=22):
#     '''
#     - tensor : [bs, seq_len, feat_dim]
#     '''
#     bs, seq_len = tensor.shape[:2]
#     tensor_ = tensor.reshape(bs, seq_len, n_players, -1) # [bs, seq_len, n_agents, x_dim]

#     x_tensor = tensor_[..., 0].unsqueeze(-1) # [bs, seq_len, n_agents, 1]
#     y_tensor = tensor_[..., 1].unsqueeze(-1)
#     xy_tensor = torch.cat([x_tensor, y_tensor], dim=-1) # [bs, seq_len, n_agents, 2]
#     xy_sum_tensor = torch.sum(xy_tensor, dim=-1) # [bs, seq_len, n_agents]

#     sorted_tensor = tensor_.clone()
#     for batch in range(bs):
#         sort_indices = torch.argsort(xy_sum_tensor[batch, 0], dim=0) # [n_agents]
#         sorted_tensor[batch] = tensor_[batch, :, sort_indices]

#     return sorted_tensor.flatten(2,3) # [bs, seq_len, feat_dim]

def xy_sort_tensor(tensor, sort_idxs_tensor=None, n_players=22, mode="sort"):
    '''
    - tensor : [bs, seq_len, feat_dim]
    '''
    bs, seq_len = tensor.shape[:2]
    tensor_ = tensor.reshape(bs, seq_len, n_players, -1) # [bs, seq_len, n_agents, x_dim]

    x_tensor = tensor_[..., 0].unsqueeze(-1) # [bs, seq_len, n_agents, 1]
    y_tensor = tensor_[..., 1].unsqueeze(-1)

    xy_tensor = torch.cat([x_tensor, y_tensor], dim=-1) # [bs, seq_len, n_agents, 2]
    if mode == "sort":
        xy_sum_tensor = torch.sum(xy_tensor, dim=-1) # [bs, seq_len, n_agents]

        sorted_tensor = tensor_.clone()
        sort_idxs_tensor = torch.zeros(bs, n_players, dtype=int)
        for batch in range(bs):
            sorted_idxs = torch.argsort(xy_sum_tensor[batch, 0], dim=0) # [n_agents]

            sorted_tensor[batch] = tensor_[batch, :, sorted_idxs]
            sort_idxs_tensor[batch] = sorted_idxs

        return sorted_tensor.flatten(2,3), sort_idxs_tensor # [bs, seq_len, feat_dim]
    else: # Restore sorted tensor
        assert sort_idxs_tensor is not None

        restored_tensor = tensor_.clone() # [bs, seq_len, n_agents, x_dim]
        for batch in range(bs):
            restore_indices = torch.argsort(sort_idxs_tensor[batch])
            restored_tensor[batch] = tensor_[batch, :, restore_indices, :]

        return restored_tensor.flatten(2,3) # [bs, seq_len, feat_dim]

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total

def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        if arg == "missing_pattern":
            continue
        elif arg.startswith("n_") or arg.endswith("_dim") or arg.endswith("size") or arg.endswith("_int"):
            parser.add_argument("--" + arg, type=int, required=True)
        elif arg == "dropout" or arg.endswith("_float"):
            parser.add_argument("--" + arg, type=float, required=False, default=0)
        else:
            parser.add_argument("--" + arg, action="store_true", default=False)

    args, _ = parser.parse_known_args()
    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params

def get_params_str(model_args, params):
    ret = ""
    for arg in model_args:
        if arg in params:
            ret += " {} {} |".format(arg, params[arg])
    return ret[1:-2]

def calc_coherence_loss(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor, m: torch.Tensor, add_va=True):
    p_next = p.roll(-1, dims=1)  # [bs, time, agents, 2]
    v_next = v.roll(-1, dims=1)  # [bs, time, agents, 2]
    pv_loss = torch.norm(((1 - m) * (p_next - p - v_next * 0.1))[:, :-1], p=1, dim=(1, 3))  # [bs, agents]
    if add_va:
        va_loss = torch.norm(((1 - m) * (v_next - v - a * 0.1))[:, :-1], p=1, dim=(1, 3))  # [bs, agents]
        return torch.mean((pv_loss + 0.1 * va_loss) / ((1 - m).sum(dim=1).squeeze(-1) + 1e-5))
    else:
        return torch.mean(pv_loss / ((1 - m).sum(dim=1).squeeze(-1) + 1e-5))

def calc_trace_dist(pred_tensor, target_tensor, mask_tensor, n_features=None, rescale=True, aggfunc="mean", dataset="soccer"):
    pred_xy = reshape_tensor(pred_tensor, rescale=rescale, n_features=n_features, dataset=dataset)
    target_xy = reshape_tensor(target_tensor, rescale=rescale, n_features=n_features, dataset=dataset)
    mask_xy = reshape_tensor(mask_tensor, n_features=n_features, dataset=dataset)

    if aggfunc == "mean":
        return (torch.norm((pred_xy - target_xy) * (1-mask_xy), dim=-1).sum() / ((1-mask_xy).sum() / 2))
    elif aggfunc == "tensor":
        return torch.norm((pred_xy - target_xy) * (1-mask_xy), dim=-1)
    else:  # if aggfunc == "sum"
        return torch.norm((pred_xy - target_xy) * (1-mask_xy), dim=-1).sum()

def calc_class_acc(pred_poss, target_poss, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().sum().item()

def calc_f1_score(target, pred):
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    precision = (tp / (tp + fp + 1e-6))
    recall = (tp / (tp + fn + 1e-6))
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

def calc_auc_score(target, pred):
    fpr, tpr, _ = roc_curve(target, pred, pos_label=1)
    return auc(fpr, tpr)

def step_changes_path_len_err(pred_traces, target_traces, mask, dataset="soccer"):
    '''
    pred_traces : [time, n_players * 2]
    target_traces : [time, n_players * 2]
    mask : [time, n_players * 2]
    '''
    pred_traces = reshape_tensor(pred_traces, dataset=dataset) # [time, n_players, 2]
    target_traces = reshape_tensor(target_traces, dataset=dataset)
    mask = reshape_tensor(mask, dataset=dataset)

    pred_traces = normalize_tensor(pred_traces, mode="reverse", dataset=dataset)
    target_traces = normalize_tensor(target_traces, mode="reverse", dataset=dataset)

    pred_traces = mask * target_traces + (1 - mask) * pred_traces
    
    step_size = torch.norm(pred_traces[1:] - pred_traces[:-1], dim=-1) # [time, n_players]
    pred_change_of_step_size = step_size.std(0) # [n_players]

    pred_path_length = step_size.sum(0)

    step_size = torch.norm(target_traces[1:] - target_traces[:-1], dim=-1) # [time, n_players]
    target_change_of_step_size = step_size.std(0) # [n_players]
    target_path_length = step_size.sum(0)

    change_of_step_size_err = torch.abs(pred_change_of_step_size - target_change_of_step_size).sum().item()
    path_len_err = torch.abs(pred_path_length - target_path_length).sum().item()

    return change_of_step_size_err, path_len_err

def calc_speed_err(pred_traces, target_traces, mask, dataset="soccer"):
    '''
    pred_traces : [time, n_players * 2]
    target_traces : [time, n_players * 2]
    mask : [time, n_players * 2]
    '''
    pred_traces = normalize_tensor(pred_traces, mode="reverse", dataset=dataset)
    target_traces = normalize_tensor(target_traces, mode="reverse", dataset=dataset)

    pred_traces = mask * target_traces + (1 - mask) * pred_traces

    vx_diff = torch.diff(pred_traces[..., 0::2], dim=0) / 0.1
    vy_diff = torch.diff(pred_traces[..., 1::2], dim=0) / 0.1
    pred_speed = torch.sqrt(vx_diff ** 2 + vy_diff ** 2)

    vx_diff = torch.diff(target_traces[..., 0::2], dim=0) / 0.1
    vy_diff = torch.diff(target_traces[..., 1::2], dim=0) / 0.1
    target_speed = torch.sqrt(vx_diff ** 2 + vy_diff ** 2)

    speed_err = torch.abs(pred_speed - target_speed).sum().item()

    return speed_err

def calc_statistic_metrics(pred_traces, target_traces, mask, ret, imputer, dataset="soccer"):
    '''
    pred_traces : [time, x_dim(n_players * n_features)]
    target_traces : [time, x_dim]
    mask : [time, x_dim]
    '''
    pred_traces_ = reshape_tensor(pred_traces, dataset=dataset).reshape(pred_traces.shape[0], -1)
    target_traces_ = reshape_tensor(target_traces, dataset=dataset).reshape(pred_traces.shape[0], -1)
    mask_ = reshape_tensor(mask, dataset=dataset).reshape(pred_traces.shape[0], -1)

    masked_traces = pd.DataFrame(pred_traces_ * mask_).replace(0, np.NaN)
    if imputer == "linear":
        interp = torch.tensor(masked_traces.copy(deep=True).interpolate().values)
    elif imputer == "knn":
        interp = torch.tensor(KNNImputer(n_neighbors=5, weights="distance").fit_transform(masked_traces.copy(deep=True)))
    elif imputer == "forward":
        interp = torch.tensor(masked_traces.copy(deep=True).ffill(axis=0).bfill(axis=0).values)
    elif imputer in ["target", "pred", "physics_f", "physics_b", "static_hybrid", "static_hybrid2", "train_hybrid"]:
        interp = pred_traces_

    if not imputer in ["pred", "physics_f", "physics_b", "static_hybrid", "static_hybrid2", "train_hybrid"]:
        ret[f"{imputer}_dist"] = calc_trace_dist(interp, target_traces_, mask_, aggfunc="sum", dataset=dataset)
        ret[f"{imputer}_pred"] = interp.clone()

    speed_err= calc_speed_err(interp.clone(), target_traces_, mask_, dataset)
    total_change_of_step_size, path_length = step_changes_path_len_err(interp, target_traces_, mask_, dataset)

    ret[f"{imputer}_speed"] = speed_err
    ret[f"{imputer}_change_of_step_size"] = total_change_of_step_size
    ret[f"{imputer}_path_length"] = path_length

    return ret

def reshape_tensor(tensor, rescale=False, n_features=None, mode="xy", dataset="soccer"):
    players, ps = get_dataset_config(dataset)

    tensor = tensor.clone()
    feat_dim = tensor.shape[-1] // players if n_features == None else n_features
    idx_map = {
        "xy":  (0, 1),
        "vel": (2, 3),
        "speed": (4,),
        "accel": (5,),
        "cartesian_accel": (4, 5)
    }
    idx = idx_map.get(mode, None)

    assert idx is not None, f"Invalid mode name : {mode}"
    tensor_x = tensor[..., idx[0]::feat_dim, None]
    tensor_y = tensor[..., idx[1]::feat_dim, None] if len(idx) == 2 else None

    if rescale:
        tensor_x *= ps[0]
        tensor_y *= ps[1]

    reshape_tensor = torch.cat([tensor_x, tensor_y], dim=-1) if tensor_y is not None else tensor_x

    return reshape_tensor

def normalize_tensor(tensor, mode="reverse", dataset="soccer"):
    tensor = tensor.clone()

    players, ps = get_dataset_config(dataset)

    n_features = tensor.shape[-1] // players
    if tensor.shape[-1] < players:
        n_features = tensor.shape[-1]

    if mode == "reverse":
        tensor[..., 0::n_features] *= ps[0]
        tensor[..., 1::n_features] *= ps[1]
    else: # normalization
        tensor[..., 0::n_features] /= ps[0]
        tensor[..., 1::n_features] /= ps[1]
    
    return tensor

def nll_gauss(pred_mean, pred_std, gt, eps=1e-6):
    normal_distri = torch.distributions.Normal(pred_mean, pred_std + eps)
    LL = normal_distri.log_prob(gt)
    NLL = - LL.sum(-1).mean()
    return NLL

def sample_gauss(pred_mean, pred_std, eps=1e-6):
    normal_distri = torch.distributions.Normal(pred_mean, pred_std + eps)
    return normal_distri.sample()

def load_pretrained_model(model, params, freeze=False, trial_num=301):
    save_path = f"saved/{trial_num:03d}"
    pre_trained_state_dict = torch.load(f"{save_path}/model/{params['model']}_state_dict_best.pt", map_location="cuda:0")
    new_model_state_dict = model.state_dict()
    
    sub_module_keys = []
    for k, v in pre_trained_state_dict.items():
        if k in new_model_state_dict:
            new_model_state_dict[k] = v
            sub_module_keys.append(k)
    
    if freeze:
        for k, p in model.named_parameters():
            if k in sub_module_keys:
                p.requires_grad = False
            
    model.load_state_dict(new_model_state_dict)

    print(f"successfully load pretrained submodule parameters.{trial_num}")
    return model

def compute_polygon_loc(ball_loc, camera_info=(0, -20, 20, 30), pitch_size=(108, 72)):
    '''
    ball_loc : [bs, time, 2]
    '''
    # Camera info
    camera_x = camera_info[0]
    camera_y = camera_info[1]
    camera_z = camera_info[2]
    camera_fov = camera_info[3]
    
    # Ball locations
    ball_x = ball_loc[..., 0]
    ball_y = ball_loc[..., 1]

    # Camera settings
    camera_x = pitch_size[0] / 2
    camera_loc = torch.tensor([camera_x, camera_y])
    camera_height = camera_z
    camera_ratio = (16, 9)

    camera_fov_x = math.radians(camera_fov)
    camera_fov_y = math.radians(camera_fov / camera_ratio[0] * camera_ratio[1])

    ball_right = ball_loc[..., 0] > pitch_size[0] / 2

    # Camera-ball angle
    ball_camera_dist = torch.norm(ball_loc - camera_loc, dim=-1) # [bs, time]
    camera_ball_angle = math.pi/2 - np.arctan(abs(camera_loc[1] - ball_loc[..., 1]) / abs(camera_loc[0] - ball_loc[..., 0]))
    camera_ball_angle_y = np.arctan(camera_height / ball_camera_dist)

    front_dist = camera_height / np.tan(camera_ball_angle_y + camera_fov_y / 2)
    rear_dist = camera_height / np.tan(camera_ball_angle_y - camera_fov_y / 2)    
    # front_dist = camera_height / math.tan(camera_ball_angle_y + camera_fov_y / 2)
    # rear_dist = camera_height / math.tan(camera_ball_angle_y - camera_fov_y / 2)
    
    front_ratio = front_dist / ball_camera_dist
    rear_ratio = rear_dist / ball_camera_dist

    camera_fov = math.radians(camera_fov) # in degree
    
    # Create a Polygon from the coordinates
    # ball_fov_dist_x = (ball_camera_dist * math.tan(camera_fov_x / 2)) * math.cos(camera_ball_angle)
    # ball_fov_dist_y = (-1) ** (ball_right) * (ball_camera_dist * math.tan(camera_fov_x / 2)) * math.sin(camera_ball_angle)
    # ball_fov_dist_y = (ball_camera_dist * math.tan(camera_fov / 2)) * math.sin(camera_ball_angle)
    
    ball_camera_close_dist = ball_camera_dist * front_ratio
    ball_camera_far_dist = ball_camera_dist * rear_ratio
    
    ball_fov_close_dist_x = (ball_camera_close_dist * np.tan(camera_fov_x / 2)) * np.cos(camera_ball_angle)
    ball_fov_close_dist_y = (-1) ** (ball_right) * (ball_camera_close_dist * np.tan(camera_fov_x / 2)) * np.sin(camera_ball_angle)
    
    ball_fov_far_dist_x = (ball_camera_far_dist * np.tan(camera_fov_x / 2)) * np.cos(camera_ball_angle)
    ball_fov_far_dist_y = (-1) ** (ball_right) * (ball_camera_far_dist * np.tan(camera_fov_x / 2)) * np.sin(camera_ball_angle)

    front_ratio = front_ratio.unsqueeze(-1)
    rear_ratio = rear_ratio.unsqueeze(-1)

    close_fov_center_point = ball_loc * (front_ratio) + camera_loc * (1 - front_ratio)
    far_fov_center_point = ball_loc * rear_ratio + camera_loc * (-rear_ratio + 1)

    poly_loc0 = (far_fov_center_point[..., 0] - ball_fov_far_dist_x, far_fov_center_point[..., 1] - ball_fov_far_dist_y)
    poly_loc1 = (far_fov_center_point[..., 0] + ball_fov_far_dist_x, far_fov_center_point[..., 1] + ball_fov_far_dist_y)
    poly_loc2 = (close_fov_center_point[..., 0] + ball_fov_close_dist_x, close_fov_center_point[..., 1] + ball_fov_close_dist_y)
    poly_loc3 = (close_fov_center_point[..., 0] - ball_fov_close_dist_x, close_fov_center_point[..., 1] - ball_fov_close_dist_y)

    polygon_points = (poly_loc0, poly_loc1, poly_loc2, poly_loc3)

    return polygon_points

def print_helper(ret, model_keys, trial=-1, dataset="soccer", save_txt=False):
    def get_key_index(key, model_keys):
        for model_key in model_keys:
            if model_key in key:
                return model_keys.index(model_key)

    n_players, _ = get_dataset_config(dataset)

    if save_txt:
        f = open(f"saved/{trial:03d}/results.txt", "w+")
    keys = [key for key in list(ret.keys()) if key not in ["n_frames", "n_missings"]]
    keys = sorted(keys, key=lambda x: get_key_index(x, model_keys=model_keys))
    for key in keys:
        if "df" in key:
            continue
        elif "travel" in key:
            print(f'{key} : {round(ret[key] / (ret["n_frames"] * n_players), 8)}')
        elif "change_of_step_size" in key:
            print(f'{key} : {round(ret[key] / (ret["n_frames"] * n_players), 8)}')
        elif "path_length" in key:
            print(f'{key} : {round(ret[key] / (ret["n_frames"] * n_players), 8)}')
        elif "dist" in key:
            print(f'{key} : {round(ret[key].item() / ret["n_missings"], 8)}')
        else:
            print(f'{key} : {round(ret[key] / ret["n_missings"], 8)}')
    
        if save_txt:
            if "df" in key:
                continue
            elif "travel" in key:
                f.write(f'{key} : {round(ret[key] / (ret["n_frames"] * n_players), 8)} \n')
            elif "change_of_step_size" in key:
                f.write(f'{key} : {round(ret[key] / (ret["n_frames"] * n_players), 8)} \n')
            elif "path_length" in key:
                f.write(f'{key} : {round(ret[key] / (ret["n_frames"] * n_players), 8)} \n')
            elif "dist" in key:
                f.write(f'{key} : {round(ret[key].item() / ret["n_missings"], 8)} \n')
            else:
                f.write(f'{key} : {round(ret[key] / ret["n_missings"], 8)} \n')
    if save_txt:
        f.close()
