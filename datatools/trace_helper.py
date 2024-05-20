import os
import sys
from typing import Dict, List

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import random
from collections import Counter

import matplotlib.colors as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
from matplotlib import animation
from scipy.ndimage import shift
from sklearn.impute import KNNImputer
from tqdm import tqdm

from dataset import SportsDataset
from models.brits.brits import BRITS
from models.dbhp.dbhp import DBHP
from models.graph_imputer.graph_imputer import BidirectionalGraphImputer
from models.naomi.naomi import NAOMI
from models.nrtsi.nrtsi import NRTSI
from models.utils import *


class TraceHelper:
    def __init__(self, traces: pd.DataFrame, events: pd.DataFrame = None, pitch_size: tuple = (108, 72)):
        self.traces = traces.dropna(axis=1, how="all").copy()
        self.pass_triple = []  # (pass start index, target, pred)
        self.events = events
        self.pitch_size = pitch_size

        self.team1_players = [c[:-2] for c in self.traces.columns if c.startswith("A") and c.endswith("_x")]
        self.team2_players = [c[:-2] for c in self.traces.columns if c.startswith("B") and c.endswith("_x")]

        self.team1_cols = np.array([TraceHelper.player_to_cols(p) for p in self.team1_players]).flatten().tolist()
        self.team2_cols = np.array([TraceHelper.player_to_cols(p) for p in self.team2_players]).flatten().tolist()

        self.phase_records = None

    @staticmethod
    def player_to_cols(p):
        return [f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy", f"{p}_speed", f"{p}_accel", f"{p}_ax", f"{p}_ay"]

    def calc_single_player_running_features(self, p: str, remove_outliers=True, smoothing=True, fm_data=False):
        if remove_outliers:
            MAX_SPEED = 12
            MAX_ACCEL = 8

        if smoothing:
            W_LEN = 15 if fm_data else 11
            P_ORDER = 2

        x = self.traces[f"{p}_x"].dropna()
        y = self.traces[f"{p}_y"].dropna()

        if smoothing and fm_data:
            x = pd.Series(signal.savgol_filter(x, window_length=W_LEN, polyorder=P_ORDER))
            y = pd.Series(signal.savgol_filter(y, window_length=W_LEN, polyorder=P_ORDER))

        vx = np.diff(x.values, prepend=x.iloc[0]) / 0.1
        vy = np.diff(y.values, prepend=y.iloc[0]) / 0.1

        if remove_outliers:
            speeds = np.sqrt(vx**2 + vy**2)
            is_speed_outlier = speeds > MAX_SPEED
            is_accel_outlier = np.abs(np.diff(speeds, append=speeds[-1]) / 0.1) > MAX_ACCEL
            is_outlier = is_speed_outlier | is_accel_outlier | shift(is_accel_outlier, 1, cval=True)
            vx = pd.Series(np.where(is_outlier, np.nan, vx)).interpolate(limit_direction="both").values
            vy = pd.Series(np.where(is_outlier, np.nan, vy)).interpolate(limit_direction="both").values

        if smoothing:
            vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
            vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)

        speeds = np.sqrt(vx**2 + vy**2)
        accels = np.diff(speeds, append=speeds[-1]) / 0.1

        ax = np.diff(vx, append=vx[-1]) / 0.1
        ay = np.diff(vy, append=vy[-1]) / 0.1

        if smoothing:
            accels = signal.savgol_filter(accels, window_length=W_LEN, polyorder=P_ORDER)
            ax = signal.savgol_filter(ax, window_length=W_LEN, polyorder=P_ORDER)
            ay = signal.savgol_filter(ay, window_length=W_LEN, polyorder=P_ORDER)

        if fm_data:
            self.traces.loc[:, TraceHelper.player_to_cols(p)] = (
                np.stack([x, y, vx, vy, speeds, accels, ax, ay]).round(6).T
            )
        else:  # e.g. Metrica
            self.traces.loc[x.index, TraceHelper.player_to_cols(p)[2:]] = (
                np.stack([vx, vy, speeds, accels, ax, ay]).round(6).T
            )

    def calc_running_features(self, remove_outliers=True, smoothing=True, fm_data=False):
        for p in self.team1_players + self.team2_players:
            self.calc_single_player_running_features(p, remove_outliers, smoothing, fm_data=fm_data)

        data_cols = self.team1_cols + self.team2_cols
        if "ball_x" in self.traces.columns:
            data_cols += ["ball_x", "ball_y"]
        meta_cols = self.traces.columns[: len(self.traces.columns) - len(data_cols)].tolist()
        self.traces = self.traces[meta_cols + data_cols]

    def find_anomaly_episode(self, threshold=3.0):
        """
        This function detect anomaly episode.
        If any players move over the threshold distance within a one frame, change the episode number into 0.
        (Episode number 0 is not able to use for training data.)
        """
        xy_cols = [f"{p}{t}" for p in self.team1_players + self.team2_players for t in ["_x", "_y"]]

        traces = self.traces[xy_cols].values

        frame_diff = np.diff(traces, axis=0)
        frame_diff_dist = np.sqrt(frame_diff[:, 0::2] ** 2 + frame_diff[:, 1::2] ** 2)

        if (frame_diff_dist > threshold).sum():
            self.traces.loc[:, "episode"] = 0

    @staticmethod
    def ffill_transition(team_poss):
        team = team_poss.iloc[0]
        nans = Counter(team_poss)[0] // 2
        team_poss.iloc[:-nans] = team
        return team_poss.replace({0: np.nan, 1: "A", 2: "B"})

    def find_gt_team_poss(self, player_poss_col="player_poss"):
        self.traces["team_poss"] = self.traces[player_poss_col].fillna(method="bfill").fillna(method="ffill")
        self.traces["team_poss"] = self.traces["team_poss"].apply(lambda x: x[0])

        # team_poss_dict = {"T": np.nan, "O": 0, "A": 1, "B": 2}
        # team_poss = self.traces[player_poss_col].fillna("T").apply(lambda x: x[0]).map(team_poss_dict)
        # poss_ids = (team_poss.diff().fillna(0) * team_poss).cumsum()
        # team_poss = team_poss.groupby(poss_ids, group_keys=True).apply(TraceHelper.ffill_transition)
        # team_poss = team_poss.reset_index(level=0, drop=True)
        # self.traces["team_poss"] = team_poss.fillna(method="bfill").fillna(method="ffill")

    def estimate_naive_team_poss(self):
        xy_cols = [f"{p}{t}" for p in self.team1_players + self.team2_players for t in ["_x", "_y"]]
        team_poss = pd.Series(index=self.traces.index, dtype=str)

        for phase in self.traces["phase"].unique():
            # if type(phase) == str:  # For GPS-event traces, ignore phases with n_players < 22
            #     phase_tuple = [int(i) for i in phase[1:-1].split(",")]
            #     if phase_tuple[0] < 0 or phase_tuple[1] < 0:
            #         continue

            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_gks = SportsDataset.detect_goalkeepers(phase_traces)
            team1_code, team2_code = phase_gks[0][0], phase_gks[1][0]

            ball_in_left = phase_traces[xy_cols].mean(axis=1) < self.pitch_size[0] / 2
            team_poss.loc[phase_traces.index] = np.where(ball_in_left, team1_code, team2_code)

        return team_poss

    @staticmethod
    def predict_episode(
        model: DBHP,
        dataset_type: str,
        player_traces: torch.Tensor,
        ball_traces: torch.Tensor,
        pred_keys: list = None,  # ["pred", "hybrid_s", "hybrid_s2", "hybrid_d", "linear", "knn", "ffill"]
        window_size=200,
        min_window_size=100,
        naive_baselines=False,
        gap_models=None,  # For NRTSI model
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        model_type = model.params["model"]
        device = next(model.parameters()).device

        player_traces = player_traces.unsqueeze(0).to(device)  # [1, time, x]
        # target_traces = input_traces.clone()
        if dataset_type == "soccer":
            ball_traces = ball_traces.unsqueeze(0).to(device)  # [1, time, 2]

        if dataset_type == "afootball":
            in_dim = model.params["team_size"] * model.params["n_features"]
            # in_dim = model.params["team_size"] * 6
            player_out_dim = model.params["n_features"]
            # player_out_dim = 6 if model.params["cartesian_accel"] else 4
            out_dim = model.params["team_size"] * player_out_dim
            out_xy_dim = model.params["team_size"] * 2
        else:
            in_dim = 2 * model.params["team_size"] * model.params["n_features"]
            player_out_dim = model.params["n_features"]
            # player_out_dim = 6 if model.params["cartesian_accel"] else 4
            out_dim = 2 * model.params["team_size"] * player_out_dim
            out_xy_dim = 2 * model.params["team_size"] * 2

        seq_len = player_traces.shape[1]

        ret = dict()
        ret["input"] = torch.zeros(seq_len, in_dim)
        ret["target"] = torch.zeros(seq_len, in_dim)
        ret["mask"] = -torch.ones(seq_len, in_dim)

        for k in pred_keys:
            if k == "pred":
                ret["pred"] = torch.zeros(seq_len, out_dim)  # [time, players * feats]
            else:
                ret[k] = torch.zeros(seq_len, out_xy_dim)  # [time, players * 2]

        if model_type == "dbhp" and model.params["dynamic_hybrid"]:
            n_lambdas = 2 if model.params["missing_pattern"] == "forecast" else 3
            if dataset_type == "afootball":
                ret["lambdas"] = torch.zeros(seq_len, model.params["team_size"] * n_lambdas)
            else:
                ret["lambdas"] = torch.zeros(seq_len, (model.params["team_size"] * 2) * n_lambdas)

        # if model.params["missing_pattern"] == "camera":
        #     n_windows = 1

        if player_traces.shape[1] % window_size < min_window_size:
            n_windows = player_traces.shape[1] // window_size
        else:
            n_windows = player_traces.shape[1] // window_size + 1

        for i in range(n_windows):
            # if model.params["missing_pattern"] == "camera":
            #     i_from = 0
            #     i_to = player_traces.shape[1]

            i_from = window_size * i
            i_to = window_size * (i + 1) if i < n_windows - 1 else player_traces.shape[1]

            window_player_traces = player_traces[:, i_from:i_to]
            if dataset_type == "soccer":  # For simulated camera view
                window_ball_traces = ball_traces[:, i_from:i_to]

            # Run model
            if dataset_type == "soccer":
                window_inputs = [window_player_traces, window_ball_traces]
            else:
                window_inputs = [window_player_traces]

            if model.params["model"] == "nrtsi":
                window_ret = model.forward(
                    window_inputs, model=model, gap_models=gap_models, mode="test", device=device
                )
            elif model.params["model"] == "graph_imputer":
                window_ret = model.evaluate(window_inputs, device=device)
            else:
                window_ret = model.forward(window_inputs, mode="test", device=device)

            # Save results for the window
            ret["input"][i_from:i_to] = window_ret["input"].detach().cpu().squeeze(0)  # [ws, x]
            ret["target"][i_from:i_to] = window_ret["target"].detach().cpu().squeeze(0)
            ret["mask"][i_from:i_to] = window_ret["mask"].detach().cpu().squeeze(0)
            for k in pred_keys:
                if k in window_ret.keys():
                    # print(k, ret[k].shape, window_ret[k].shape)
                    ret[k][i_from:i_to] = window_ret[k].detach().cpu().squeeze(0)

            if model_type == "dbhp" and model.params["dynamic_hybrid"]:
                ret["lambdas"][i_from:i_to] = window_ret["lambdas"].detach().cpu().squeeze(0)

        if naive_baselines:
            masked_traces = reshape_tensor(ret["input"], dataset_type=dataset_type).reshape(ret["input"].shape[0], -1)
            masked_traces = pd.DataFrame(masked_traces).replace(0, np.NaN)

            ret["linear"] = torch.FloatTensor(masked_traces.copy(deep=True).interpolate().values)
            knn_pred = KNNImputer(n_neighbors=5, weights="distance").fit_transform(masked_traces.copy(deep=True))
            ret["knn"] = torch.FloatTensor(knn_pred)
            ret["ffill"] = torch.FloatTensor(masked_traces.copy(deep=True).ffill(axis=0).bfill(axis=0).values)

        # Unnormalize traces
        for k in pred_keys + ["target"]:
            if model.params["normalize"]:
                ret[k] = normalize_tensor(ret[k], mode="upscale", dataset_type=dataset_type)

        stats = dict()
        stats["total_frames"] = seq_len
        stats["missing_frames"] = int(((1 - ret["mask"]).sum() / model.params["n_features"]).item())
        # stats["missing_frames"] = int(((1 - ret["mask"]).sum() / 6).item())
        for k in pred_keys:
            errors = calc_pred_errors(ret[k], ret["target"], ret["mask"], dataset_type)
            stats[f"{k}_pe"] = errors[0]  # pos_error
            stats[f"{k}_se"] = errors[1]  # speed_error
            stats[f"{k}_sce"] = errors[2]  # step_change_error
            stats[f"{k}_ple"] = errors[3]  # path_length_error
            # print(f"key : {k}, pe : {errors[0]}")
        
        return ret, stats

    def predict(
        self,
        model: DBHP,
        dataset_type="soccer",
        min_episode_size=100,
        naive_baselines=False,
        gap_models=None,
    ) -> Tuple[dict]:
        model_type = model.params["model"]
        random.seed(1000)
        np.random.seed(1000)

        players = self.team1_players + self.team2_players
        feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
        player_cols = [f"{p}{x}" for p in players for x in feature_types]

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
        ret["mask"] = pd.DataFrame(-1, index=self.traces.index, columns=player_cols)
        ret["mask"].loc[:, "episode"] = self.traces["episode"]
        for k in pred_keys:
            ret[k] = self.traces.copy(deep=True)

        if model_type == "dbhp" and model.params["dynamic_hybrid"]:
            lambda_types = ["_w0", "_w1"] if model.params["missing_pattern"] == "forecast" else ["_w0", "_w1", "_w2"]
            lambda_cols = [f"{p}{w}" for p in players for w in lambda_types]
            ret["lambdas"] = pd.DataFrame(-1, index=self.traces.index, columns=lambda_cols)

        x_cols = [c for c in self.traces.columns if c.endswith("_x")]
        y_cols = [c for c in self.traces.columns if c.endswith("_y")]

        if model.params["normalize"]:
            self.traces[x_cols] /= self.pitch_size[0]
            self.traces[y_cols] /= self.pitch_size[1]
            self.pitch_size = (1, 1)

        for phase in self.traces["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase]

            phase_gks = SportsDataset.detect_goalkeepers(phase_traces)
            team1_code, team2_code = phase_gks[0][0], phase_gks[1][0]

            phase_player_cols = phase_traces[player_cols].dropna(axis=1).columns
            team1_cols = [c for c in phase_player_cols if c.startswith(team1_code)]
            team2_cols = [c for c in phase_player_cols if c.startswith(team2_code)]
            ball_cols = ["ball_x", "ball_y"]

            # reorder teams so that the left team comes first
            phase_player_cols = team1_cols + team2_cols

            if min(len(team1_cols), len(team2_cols)) < model.params["team_size"] * len(feature_types):
                continue

            episodes = [e for e in phase_traces["episode"].unique() if e > 0]
            for e in tqdm(episodes, desc=f"Phase {phase}"):
                ep_traces = phase_traces[phase_traces["episode"] == e]
                if len(ep_traces) < min_episode_size:
                    continue

                ep_player_traces = torch.FloatTensor(ep_traces[phase_player_cols].values)
                ep_ball_traces = torch.FloatTensor(ep_traces[ball_cols].values)

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

                # Update resulting DataFrames
                pos_cols = [c for c in phase_player_cols if c[-2:] in ["_x", "_y"]]
                if model.params["cartesian_accel"]:
                    dp_cols = phase_player_cols
                else:
                    dp_cols = [c for c in phase_player_cols if c[-3:] not in ["_ax", "_ay"]]

                for k in pred_keys + ["target", "mask"]:
                    if  k in ["pred", "target", "mask"]:
                        cols = dp_cols if model_type == "dbhp" else pos_cols
                        ret[k].loc[ep_traces.index, cols] = np.array(ep_ret[k])
                        # ret[k].loc[ep_traces.index, dp_cols] = np.array(ep_ret[k])
                    elif naive_baselines and k in ["linear", "knn", "ffill"]:
                        ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])
                    else:
                        ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])

                if model_type == "dbhp" and model.params["dynamic_hybrid"]:
                    ep_players = [c[:-2] for c in phase_player_cols if "_x" in c]
                    lambda_cols = [f"{p}{w}" for p in ep_players for w in lambda_types]
                    ret["lambdas"].loc[ep_traces.index, lambda_cols] = np.array(ep_ret["lambdas"])

                for key in ep_stats:
                    stats[key] += ep_stats[key]

        # if model.params["normalize"]:
        #     self.pitch_size = (108, 72)
        #     self.traces[x_cols] *= self.pitch_size[0]
        #     self.traces[y_cols] *= self.pitch_size[1]

        return ret, stats

    @staticmethod
    def plot_speeds_and_accels(traces: pd.DataFrame, players: list = None) -> animation.FuncAnimation:
        FRAME_DUR = 30
        MAX_SPEED = 40
        MAX_ACCEL = 6

        if players is None:
            players = [c[:3] for c in traces.columns if c.endswith("_speed")]
        else:
            players = [p for p in players if f"{p}_speed" in traces.columns]
        players.sort()

        if len(players) > 20:
            print("Error: No more than 20 players")
            return

        fig, axes = plt.subplots(4, 1)
        fig.set_facecolor("w")
        fig.set_size_inches(15, 10)
        plt.rcParams.update({"font.size": 15})

        times = traces["time"].values
        t0 = int(times[0] - 0.1)

        axes[0].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, MAX_SPEED))
        axes[1].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[2].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[3].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[0].set_ylabel("speed")
        axes[1].set_ylabel("aceel")
        axes[2].set_ylabel("aceel_x")
        axes[3].set_ylabel("aceel_y")

        for i, ax in enumerate(axes):
            ax.grid()
            if len(axes) - 1 == i:
                ax.set_xlabel("time")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        speed_plots = dict()
        accel_plots = dict()
        accel_x_plots = dict()
        accel_y_plots = dict()
        colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]

        for i, p in enumerate(players):
            speeds = traces[f"{p}_speed"].values * 3.6
            accels = traces[f"{p}_accel"].values
            accels_x = traces[f"{p}_ax"].values
            accels_y = traces[f"{p}_ay"].values
            (speed_plots[p],) = axes[0].plot(times, speeds, color=colors[i], label=p)
            (accel_plots[p],) = axes[1].plot(times, accels, color=colors[i], label=p)
            (accel_x_plots[p],) = axes[2].plot(times, accels_x, color=colors[i], label=p)
            (accel_y_plots[p],) = axes[3].plot(times, accels_y, color=colors[i], label=p)

        axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))

        def animate(i):
            for ax in axes:
                ax.set_xlim(10 * i, 10 * i + FRAME_DUR)

        frames = (len(traces) - 10 * FRAME_DUR) // 100 + 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200)
        plt.close(fig)

        return anim


if __name__ == "__main__":
    import json

    from models import load_model

    trial = 810
    device = "cuda:0"
    save_path = f"saved/{trial:03d}"
    with open(f"{save_path}/params.json", "r") as f:
        params = json.load(f)
    model = load_model(params["model"], params).to(device)

    model_path = f"saved/{trial}"
    state_dict = torch.load(
        f"{model_path}/model/{params['model']}_state_dict_best.pt",
        map_location=lambda storage, _: storage,
    )
    model.load_state_dict(state_dict)

    # match_id = "20862-20875"
    # match_traces = pd.read_csv(f"data/gps_event_traces_gk_pred/{match_id}.csv", header=0, encoding="utf-8-sig")
    # helper = TraceHelper(match_traces)
    # pred_poss = helper.predict(model, split=False, evaluate=True)