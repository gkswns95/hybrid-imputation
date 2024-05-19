import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import shift
from torch.autograd import Variable
from tqdm import tqdm

from dataset import SportsDataset
from datatools.trace_helper import TraceHelper

from models.brits.brits import BRITS
from models.dbhp.dbhp import DBHP
from models.graph_imputer.graph_imputer import BidirectionalGraphImputer
from models.naomi.naomi import NAOMI
from models.nrtsi.nrtsi import NRTSI
from models.utils import *


class NFLDataHelper(TraceHelper):
    def __init__(self, pitch_size=(110, 49), n_sample=1, data_path=None, traces=None):
        if traces is not None:
            self.traces = traces
        self.data_path = data_path
        self.total_players = 6
        self.player_cols = [f"player{p}" for p in range(self.total_players)]
        self.player_xy_cols = [f"player{p}{t}" for p in range(self.total_players) for t in ["_x", "_y"]]

        self.ws = 50
        self.n_sample = n_sample
        self.pitch_size = pitch_size

    def reconstruct_df(self):
        traces_np = np.load(self.data_path)

        bs, seq_len = traces_np.shape[:2]

        # traces_np[..., :6] *= self.pitch_size[0]
        # traces_np[..., 6:] *= self.pitch_size[1]

        x_data = traces_np[..., :6, None]
        y_data = traces_np[..., 6:, None]
        xy_data = np.concatenate([x_data, y_data], axis=-1)

        traces_np = xy_data.reshape(bs, seq_len, -1)  # Rearranging the order of the x and y positions.
        traces_np = traces_np.reshape(-1, self.total_players * 2)  # [timesteps, 12]

        traces_df = pd.DataFrame(traces_np, columns=self.player_xy_cols, dtype=float)

        episodes = np.zeros(len(traces_df))
        episodes[0 :: self.ws] = 1
        episodes = episodes.cumsum()

        traces_df["episode"] = episodes.astype("int")
        traces_df["frame"] = np.arange(len(traces_df)) + 1

        self.traces = traces_df

    @staticmethod
    def player_to_cols(p):
        return [f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy", f"{p}_speed", f"{p}_accel", f"{p}_ax", f"{p}_ay"]

    def calc_single_player_running_features(
        self, p: str, episode_traces: pd.DataFrame, remove_outliers=True, smoothing=True
    ):
        episode_traces = episode_traces[[f"{p}_x", f"{p}_y"]]
        x = episode_traces[f"{p}_x"]
        y = episode_traces[f"{p}_y"]

        fps = 0.1
        if remove_outliers:
            MAX_SPEED = 12
            MAX_ACCEL = 8

        if smoothing:
            W_LEN = 12
            P_ORDER = 2
            x = pd.Series(signal.savgol_filter(x, window_length=W_LEN, polyorder=P_ORDER))
            y = pd.Series(signal.savgol_filter(y, window_length=W_LEN, polyorder=P_ORDER))

        vx = np.diff(x.values, prepend=x.iloc[0]) / fps
        vy = np.diff(y.values, prepend=y.iloc[0]) / fps

        if remove_outliers:
            speeds = np.sqrt(vx**2 + vy**2)
            is_speed_outlier = speeds > MAX_SPEED
            is_accel_outlier = np.abs(np.diff(speeds, append=speeds[-1]) / fps) > MAX_ACCEL
            is_outlier = is_speed_outlier | is_accel_outlier | shift(is_accel_outlier, 1, cval=True)

            vx = pd.Series(np.where(is_outlier, np.nan, vx)).interpolate(limit_direction="both").values
            vy = pd.Series(np.where(is_outlier, np.nan, vy)).interpolate(limit_direction="both").values

        if smoothing:
            vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
            vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)

        speeds = np.sqrt(vx**2 + vy**2)
        accels = np.diff(speeds, append=speeds[-1]) / fps

        ax = np.diff(vx, append=vx[-1]) / fps
        ay = np.diff(vy, append=vy[-1]) / fps

        if smoothing:
            accels = signal.savgol_filter(accels, window_length=W_LEN, polyorder=P_ORDER)
            ax = signal.savgol_filter(ax, window_length=W_LEN, polyorder=P_ORDER)
            ay = signal.savgol_filter(ay, window_length=W_LEN, polyorder=P_ORDER)

        self.traces.loc[episode_traces.index, NFLDataHelper.player_to_cols(p)] = (
            np.stack([x, y, vx, vy, speeds, accels, ax, ay]).round(6).T
        )

    def calc_running_features(self, remove_outliers=False, smoothing=False):
        episode = self.traces["episode"].unique()

        for e in tqdm(episode, desc="Calculating running features..."):
            episode_traces = self.traces[self.traces.episode == e]

            for p in self.player_cols:
                self.calc_single_player_running_features(
                    p, episode_traces, remove_outliers=remove_outliers, smoothing=smoothing
                )

        player_cols = np.array([NFLDataHelper.player_to_cols(p) for p in self.player_cols]).flatten().tolist()
        self.traces = self.traces[["episode", "frame"] + player_cols]  # Rearange columns.

    @staticmethod
    def predict_episode(
        input: list,
        ret_keys: list,
        model_keys: list,
        model: nn.Module,
        wlen=200,
        statistic_metrics=False,
        gap_models=None,  # For NRTSI model
        dataset="soccer",
    ) -> torch.Tensor:
        device = next(model.parameters()).device

        input_traces = input[0].unsqueeze(0).to(device)  # [1, time, x_dim]
        target_traces = input_traces.clone()
        if dataset == "soccer":
            ball_traces = input[1].unsqueeze(0).to(device)  # [1, time, 2]

        output_dim = model.params["n_features"]
        output_dim *= model.params["n_players"]
        if model.params["model"] == "nrtsi":
            output_dim *= 4

        seq_len = input_traces.shape[1]

        # Init episode ret
        episode_ret = {key: 0 for key in ret_keys}

        episode_pred = torch.zeros(seq_len, output_dim)
        episode_mask = torch.ones(seq_len, output_dim) * -1

        episode_pred_dict = {}
        for key in model_keys:
            episode_pred_dict[key] = torch.zeros(seq_len, (model.params["n_players"] * 2) * 2)  # [time, out_dim]

        for i in range(input_traces.shape[1] // wlen + 1):
            i_from = wlen * i
            i_to = wlen * (i + 1)

            window_input = input_traces[:, i_from:i_to]
            window_target = target_traces[:, i_from:i_to]
            if dataset == "soccer":
                window_ball = ball_traces[:, i_from:i_to]

            window_target_ = reshape_tensor(target_traces[:, i_from:i_to], dataset_type=dataset).flatten(2, 3)
            if window_input.shape[1] != wlen:
                for key in model_keys:
                    episode_pred_dict[key][i_from:i_to] = window_target_
                continue

            # Run model
            if dataset == "soccer":
                window_inputs = [window_input, window_target, window_ball]
            else:
                window_inputs = [window_input, window_target]
            if model.params["model"] == "nrtsi":
                window_ret = model.forward(
                    window_inputs, model=model, gap_models=gap_models, mode="test", device=device
                )
            else:
                window_ret = model.forward(window_inputs, mode="test", device=device)

            targets = window_ret["target"].detach().cpu().squeeze(0)
            masks = window_ret["mask"].detach().cpu().squeeze(0)

            # Compute statistic metrics
            if statistic_metrics:
                for key in model_keys:
                    if key in ["linear", "knn", "forward"]:
                        imputer_key = "target"
                    elif key in ["pred"]:
                        imputer_key = key
                    else:
                        imputer_key = key + "_pred"
                    traces = window_ret[imputer_key].detach().cpu().squeeze(0)
                    calc_pred_errors(traces, targets, masks, window_ret, pred_type=key, dataset_type=dataset)

            # Save sequence results
            episode_mask[i_from:i_to] = window_ret["mask"].detach().cpu().squeeze(0)
            for key in model_keys:
                if key == "pred":
                    episode_pred[i_from:i_to] = window_ret["pred"].detach().cpu().squeeze(0)
                else:
                    episode_pred_dict[key][i_from:i_to] = window_ret[f"{key}_pred"].detach().cpu().squeeze(0)

            for key in ret_keys:
                if key == "total_frames":
                    episode_ret[key] += seq_len
                elif key == "missing_frames":
                    episode_ret[key] += ((1 - masks).sum() / model.params["n_features"]).item()
                else:
                    episode_ret[key] += window_ret[key]

        # Update episode ret
        episode_df_ret = {"mask": episode_mask}
        for key in model_keys:
            if key == "pred":
                if model.params["normalize"]:
                    episode_pred = normalize_tensor(episode_pred, mode="upscale")
                episode_df_ret["pred"] = episode_pred
            else:
                if model.params["normalize"]:
                    episode_pred_dict[key] = normalize_tensor(episode_pred_dict[key], mode="upscale")
                episode_df_ret[f"{key}_df"] = episode_pred_dict[key]

        return episode_ret, episode_df_ret

    def predict(
        self,
        model: DBHP,
        dataset_type="afootball",
        min_episode_size=50,
        naive_baselines=False,
        gap_models=None,
    ) -> Tuple[dict]:
        model_type = model.params["model"]
        random.seed(1000)
        np.random.seed(1000)

        feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
        player_cols = [f"player{p}{x}" for p in range(6) for x in feature_types]

        pred_keys = ["pred"]
        if model_type == "dbhp":
            if model.params["deriv_accum"]:
                pred_keys += ["dap_f"]
                if model.params["missing_pattern"] != "forecast":
                    pred_keys += ["dap_b"]
            if model.params["dynamic_hybrid"]:
                if model.params["missing_pattern"] == "forecast":
                    pred_keys += ["hybrid_d"]
                else:
                    pred_keys += ["hybrid_s", "hybrid_s2", "hybrid_d"]
        if naive_baselines:
            if model.params["missing_pattern"] == "forecast":
                pred_keys += ["ffill"]
            else:
                pred_keys += ["linear", "knn", "ffill"]

        stat_keys = ["total_frames", "missing_frames"]
        stat_keys += [f"{k}_{m}" for k in pred_keys for m in ["pe", "se", "sce", "ple"]]

        stats = {k: 0 for k in stat_keys}

        # initialize resulting DataFrames
        ret = dict()
        ret["target"] = self.traces.copy(deep=True)
        ret["mask"] = pd.DataFrame(-1, index=self.traces.index, columns=["episode"] + player_cols)
        ret["mask"].loc[:, "episode"] = self.traces["episode"]
        for k in pred_keys:
            ret[k] = self.traces.copy(deep=True)

        if model_type == "dbhp" and model.params["dynamic_hybrid"]:
            lambda_types = ["_w0", "_w1"] if model.params["missing_pattern"] == "forecast" else ["_w0", "_w1", "_w2"]
            lambda_cols = [f"{p}{w}" for p in range(10) for w in lambda_types]
            ret["lambdas"] = pd.DataFrame(-1, index=self.traces.index, columns=lambda_cols)

        x_cols = [c for c in self.traces.columns if c.endswith("_x")]
        y_cols = [c for c in self.traces.columns if c.endswith("_y")]

        if model.params["normalize"]:
            self.traces[x_cols] /= self.pitch_size[0]
            self.traces[y_cols] /= self.pitch_size[1]
            self.pitch_size = (1, 1)

        episodes = [e for e in self.traces["episode"].unique() if e > 0]
        for episode in tqdm(episodes, desc="Episode"):
            ep_traces = self.traces[self.traces["episode"] == episode]
            if len(ep_traces) < min_episode_size:
                continue
            ep_player_traces = torch.FloatTensor(ep_traces[player_cols].values)
            ep_ball_traces = ep_player_traces.clone()
            with torch.no_grad():
                    ep_ret, ep_stats = TraceHelper.predict_episode(
                        model,
                        dataset_type,
                        ep_player_traces,
                        ep_ball_traces,
                        pred_keys=pred_keys,
                        window_size=model.params["window_size"],
                        min_window_size=min_episode_size,
                        naive_baselines=naive_baselines,
                        gap_models=gap_models,
                    )

            # update resulting DataFrames
            pos_cols = [c for c in player_cols if c[-2:] in ["_x", "_y"]]
            if model.params["cartesian_accel"]:
                dp_cols = player_cols
            else:
                dp_cols = [c for c in player_cols if c[-3:] not in ["_ax", "_ay"]]

            for k in pred_keys + ["target", "mask"]:
                if k in ["pred", "target", "mask"]:
                    cols = dp_cols if model_type == "dbhp" else pos_cols
                    ret[k].loc[ep_traces.index, cols] = np.array(ep_ret[k])
                    # ret[k].loc[ep_traces.index, dp_cols] = np.array(ep_ret[k])
                elif naive_baselines and k in ["linear", "knn", "ffill"]:
                    ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])
                else:
                    ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])

            if model_type == "dbhp" and model.params["dynamic_hybrid"]:
                ep_players = [c[:-2] for c in player_cols if "_x" in c]
                lambda_cols = [f"{p}{w}" for p in ep_players for w in lambda_types]
                ret["lambdas"].loc[ep_traces.index, lambda_cols] = np.array(ep_ret["lambdas"])

            for key in ep_stats:
                stats[key] += ep_stats[key]

        return ret, stats

    def predict2(self, trial: int, model: nn.Module):
        fig_path = f"saved/{trial:03d}/impute/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        if model.params["model"] == "nrtsi":
            ckpt_path_dict = dict()
            ckpt_root_dir = f"./saved/{trial}/model/"
            ckpt_path_dict[1] = os.path.join(ckpt_root_dir, "nrtsi_state_dict_best_gap_1.pt")
            ckpt_path_dict[2] = os.path.join(ckpt_root_dir, "nrtsi_state_dict_best_gap_2.pt")
            ckpt_path_dict[4] = os.path.join(ckpt_root_dir, "nrtsi_state_dict_best_gap_4.pt")
            ckpt_path_dict[8] = os.path.join(ckpt_root_dir, "nrtsi_state_dict_best_gap_8.pt")
            ckpt_path_dict[16] = os.path.join(ckpt_root_dir, "nrtsi_state_dict_best_gap_16.pt")

            ckpt_dict = dict()
            for key in ckpt_path_dict:
                ckpt_dict[key] = torch.load(ckpt_path_dict[key])
        else:
            ckpt_dict = None

        random.seed(1234)
        torch.manual_seed(1234)

        test_dataset = SportsDataset(
            data_paths=["data/nfl_traces/nfl_test.csv"],
            target_type="imputation",
            train=False,
            load_saved=False,
            save_new=False,
            n_features=model.params["n_features"],
            cartesian_accel=model.params["cartesian_accel"],
            normalize=False,
            flip_pitch=False,
            overlap=False,
        )
        test_data = test_dataset.input_data

        loss, avg_loss, target_sc, imp_sc, gt_path_len, path_len, pos_dist = NFLDataHelper.run_imputation(
            model, test_data, ckpt_dict, fig_path, n_sample=self.n_sample, model_name=model.params["model"]
        )
        output_str = (
            "Testing_Loss: %4f, Avg Loss: %4f, Gt Change of Step Size: %4f, Impute Change of Step Size: %4f,"
            "Gt Path Len: %4f, Path Len: %4f" % (loss, avg_loss, target_sc, imp_sc, gt_path_len, path_len)
        )

        return output_str, pos_dist

    @staticmethod
    def run_imputation(
        model, exp_data, ckpt_dict, fig_path, n_sample=10, batch_size=64, save_all_imgs=False, model_name="nrtsi"
    ):

        model.eval()

        device = next(model.parameters()).device

        # if model_name == "nrtsi":
        #     imputer = NRTSIImputer(model.params).to(device=device)
        # elif model_name == "dbhp":
        #     imputer = PeImputer(model.params).to(device=device)

        inds = np.arange(exp_data.shape[0])
        i = 0
        loss = 0
        avg_loss = 0
        count = 0

        pos_dist = 0.0
        missing_frames = 0.0

        total_sc = 0
        target_total_sc = 0
        path_length = 0
        target_path_length = 0

        if exp_data.shape[0] < batch_size:
            batch_size = exp_data.shape[0]
        while i + batch_size <= exp_data.shape[0]:
            print(f"[{i + batch_size}/{exp_data.shape[0]}]")
            ind = torch.from_numpy(inds[i : i + batch_size]).long()
            i += batch_size
            data = exp_data[ind]

            # Randomly permute player order for nfl
            data, _ = shuffle_players(data, 6)

            data = data.to(device)
            ground_truth = data.clone()

            # Change (batch, time, x) to (time, batch, x)
            data = Variable(data.transpose(0, 1))
            ground_truth = ground_truth.transpose(0, 1)

            seq_len, bs = data.shape[:2]

            min_mse = 1e5 * np.ones(batch_size)
            avg_mse = np.zeros(batch_size)

            mask = np.ones((seq_len, 6))
            num_missing = random.randint(40, 49)
            missing_list_np = np.array(random.sample(range(seq_len), num_missing))
            mask[missing_list_np] = 0.0
            mask[0] = 1.0

            time_gap = time_interval(mask, list(range(seq_len)))
            time_gap = torch.tensor(time_gap, dtype=torch.float32).unsqueeze(0)
            time_gap = torch.repeat_interleave(time_gap, model.params["n_features"], dim=-1).expand(bs, -1, -1)

            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_agents]
            mask = torch.repeat_interleave(mask, model.params["n_features"], dim=-1).expand(
                bs, -1, -1
            )  # [bs, time, feat_dim]

            if model_name == "nrtsi":
                for d in range(n_sample):
                    imputation = ground_truth.clone()
                    obs_list_np = sorted(list(set(np.arange(data.shape[0])) - set(missing_list_np)))
                    init_obs_list = obs_list = torch.from_numpy(np.array(obs_list_np)).long().to(device)
                    init_obs_data = obs_data = data[obs_list]
                    while len(obs_list_np) < data.shape[0]:
                        next_list_np, gap = NFLDataHelper.get_next_to_impute(data.shape[0], obs_list_np, max_level=4)
                        if gap > 2**2:  # section 3.3 in NRTSI paper(Stochastic Time Series)
                            next_list_np = [next_list_np[0]]

                        max_gap = NFLDataHelper.gap_to_max_gap(gap)
                        model.load_state_dict(ckpt_dict[max_gap])
                        obs_list = torch.from_numpy(np.array(obs_list_np)).long().to(device)
                        obs_list_np += next_list_np

                        obs_list = obs_list[None, :, None].repeat(batch_size, 1, 1)  # [batch_size, time, 1]
                        next_list = torch.from_numpy(np.array(next_list_np)).long().to(device)
                        next_list = next_list[None, :, None].repeat(batch_size, 1, 1)  # [batch_size, time, 1]

                        with torch.no_grad():
                            prediction = model.forward2(obs_data.transpose(0, 1), obs_list, next_list, gap)

                        samples = NFLDataHelper.sample_gauss(prediction, ground_truth, gap)

                        imputation[next_list_np] = samples.transpose(0, 1)
                        obs_data = torch.cat([obs_data, samples.transpose(0, 1)], 0)

                    """
                    imputation : [time, bs, x_dim]
                    ground_truth : [time, bs, x_dim]
                    mask : [time, bs, x_dim]
                    """
                    imputation_ = reshape_tensor(imputation, upscale=False, dataset_type="football")  # [time, bs, 6, 2]
                    ground_truth_ = reshape_tensor(ground_truth, upscale=False, dataset_type="football")

                    step_size = torch.norm(imputation_[1:] - imputation_[:-1], dim=-1)
                    total_sc += step_size.std(0).mean()
                    path_length += step_size.sum(0).mean()

                    step_size = torch.norm(ground_truth_[1:] - ground_truth_[:-1], dim=-1)
                    target_total_sc += step_size.std(0).mean().item()
                    target_path_length += step_size.sum(0).mean()

                    imputation = imputation_.flatten(2, 3)  # [time, bs, -1]
                    ground_truth = ground_truth_.flatten(2, 3)

                    mse = (torch.sum((imputation - ground_truth).pow(2), [0, 2]) / num_missing).cpu().numpy()
                    avg_mse += mse
                    min_mse[mse < min_mse] = mse[mse < min_mse]

                if torch.cuda.is_available():
                    mask = mask.to(device)
                mask = mask.transpose(0, 1)  # [time, bs, feat_dim]

            elif model_name in ["dbhp", "brits", "naomi"]:
                input_dict = {}

                # data[missing_list_np] = 0.0 # masking missing values

                data = data.transpose(0, 1)  # [bs, time, x_dim]
                ground_truth = ground_truth.transpose(0, 1)

                if torch.cuda.is_available():
                    mask, data, ground_truth, time_gap = (
                        mask.to(device),
                        data.to(device),
                        ground_truth.to(device),
                        time_gap.to(device),
                    )

                data = data * mask

                for d in range(n_sample):
                    input_dict["input"] = data.clone()
                    input_dict["target"] = ground_truth.clone()
                    input_dict["mask"] = mask.clone()
                    input_dict["delta"] = time_gap.clone()

                    with torch.no_grad():
                        ret = model.forward2(input_dict, device=device)

                    if model_name == "dbhp":
                        if model.params["dynamic_hybrid"]:
                            imputation = ret["hybrid_d"]
                        else:
                            imputation = ret["hybrid_s"]
                    else:
                        imputation = ret["pred"]

                    target_ = ret["target"]
                    mask_ = ret["mask"]

                    xy_mask = reshape_tensor(mask_, dataset_type="afootball").flatten(2, 3)
                    xy_target = reshape_tensor(target_, dataset_type="afootball").flatten(2, 3)
                    imputation = imputation * (1 - xy_mask) + xy_target * xy_mask

                    imputation = imputation.transpose(0, 1)  # [time, bs, x_dim]
                    target_ = target_.transpose(0, 1)
                    mask_ = mask_.transpose(0, 1)

                    imputation_ = reshape_tensor(
                        imputation, upscale=False, dataset_type="afootball"
                    )  # [time, bs, 6, 2]
                    target_ = reshape_tensor(target_, upscale=False, dataset_type="afootball")

                    step_size = torch.norm(imputation_[1:] - imputation_[:-1], dim=-1)
                    total_sc += step_size.std(0).mean()
                    path_length += step_size.sum(0).mean()

                    step_size = torch.norm(target_[1:] - target_[:-1], dim=-1)
                    target_total_sc += step_size.std(0).mean().item()
                    target_path_length += step_size.sum(0).mean()

                    imputation_ = imputation_.flatten(2, 3)  # [time, bs, -1]
                    target_ = target_.flatten(2, 3)

                    mse = (torch.sum((imputation_ - target_).pow(2), [0, 2]) / num_missing).cpu().numpy()
                    avg_mse += mse
                    min_mse[mse < min_mse] = mse[mse < min_mse]

                    if model_name in ["dbhp", "brits", "naomi", "nrtsi"]:
                        pos_dist += calc_pos_error(imputation_, target_, mask_, aggfunc="sum", dataset="football")
                        missing_frames += bs * num_missing * 6

            count += 1
            loss += min_mse.mean()
            avg_loss += avg_mse.mean() / n_sample

        loss /= count
        avg_loss /= count
        target_total_sc /= n_sample * count
        total_sc /= n_sample * count
        target_path_length /= n_sample * count
        path_length /= n_sample * count

        pos_dist /= missing_frames + 1e-6

        return (
            loss,
            avg_loss,
            target_total_sc,
            total_sc,
            target_path_length,
            path_length,
            pos_dist,
        )

    @staticmethod
    def get_next_to_impute(seq_len, obs_list, max_level):
        min_dist_to_obs = np.zeros(seq_len)
        for i in range(seq_len):
            if i not in obs_list:
                min_dist = np.abs((np.array(obs_list) - i)).min()
                if min_dist <= 2**max_level:
                    min_dist_to_obs[i] = min_dist
        next_idx = np.argwhere(min_dist_to_obs == np.amax(min_dist_to_obs))[:, 0].tolist()
        gap = np.amax(min_dist_to_obs)

        return next_idx, gap

    @staticmethod
    def gap_to_max_gap(gap):
        i = 0
        while gap > 2**i:
            i += 1
        return 2**i

    @staticmethod
    def sample_gauss(pred, gt, gap, eps=1e-6):
        pred_mean = pred[:, :, : gt.shape[-1]]

        pred_std = F.softplus(pred[:, :, gt.shape[-1] :]) + eps
        if gap <= 2**2:
            pred_std = 1e-5 * pred_std
        normal_distri = torch.distributions.Normal(pred_mean, pred_std)
        return normal_distri.sample()

    # @staticmethod
    # def plot_nfl(epoch, fig_path, obs_data, imputation, ground_truth, gap, d, n_sample, i=0, j=0):
    #     imputation = imputation.cpu().numpy()
    #     ground_truth = ground_truth.cpu().numpy()
    #     obs_data = obs_data.detach().cpu().numpy()
    #     plt.figure(j, figsize=(4,4))
    #     plt.xticks([])
    #     plt.yticks([])

    #     colormap = ["b", "r" , "m", "brown", "lime", "orage", "gold", "indigo", "slategrey", "y", "g"]
    #     for k in range(6):
    #         plt.plot(imputation[:,j,k], imputation[:,j,k+6], color=colormap[0], alpha=0.5, label="imputation")
    #         plt.scatter(obs_data[:,j,k], obs_data[:,j,k+6], color=colormap[-1], label="observation")

    #     for k in range(6):
    #         plt.plot(ground_truth[:,j, k], ground_truth[:,j,k+6], color=colormap[1], label="ground truth")
    #     plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     plt.legend(by_label.values(), by_label.keys(), loc="upper left")
    #     if d == (n_sample - 1):
    #         plt.savefig(os.path.join(fig_path, "test_epoch_{%d}_{%d}_{%d}_{%d}.pdf" % (epoch, gap, i, j)))
    #         plt.close()

    @staticmethod
    def plot_nfl(pred, target, obs):
        fig = plt.figure(0)
        ax = fig.add_subplot(111)

        for k in range(6):
            non_zero_indices = np.where((obs[:, 2 * k] != 0) | (obs[:, 2 * k + 1] != 0))
            ax.plot(pred[:, 2 * k], pred[:, 2 * k + 1], linestyle="--", color="blue", label="Imputation")
            # ax.scatter(pred[:,2*k], pred[:,2*k+1], color="blue", label="Imputation")
            ax.scatter(
                obs[non_zero_indices, 2 * k], obs[non_zero_indices, 2 * k + 1], color="green", label="Observation"
            )

        for k in range(6):
            ax.plot(target[:, 2 * k], target[:, 2 * k + 1], color="red", label="Ground Truth")

        plt.legend(
            handles=[
                plt.Line2D([0], [0], color="blue", linestyle="--", label="Imputation"),
                plt.Line2D([0], [0], marker="o", color="green", label="Observation", markersize=5),
                plt.Line2D([0], [0], color="red", label="Ground Truth", linestyle="-"),
            ]
        )
        plt.show()
