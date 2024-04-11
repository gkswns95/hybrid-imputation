import os
import sys

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
from tqdm import tqdm

from dataset import SportsDataset
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

        self.ws = 200
        self.ps = (108, 72)

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
        input: list,
        ret_keys: list,
        model_keys: list,
        model: nn.Module,
        wlen=200,
        statistic_metrics=False,
        gap_models=None,  # For NRTSI model
        dataset="soccer",
    ) -> torch.Tensor:
        model_name = model.params["model"]
        device = next(model.parameters()).device

        input_traces = input[0].unsqueeze(0).to(device)  # [1, time, x_dim]
        target_traces = input_traces.clone()
        if dataset == "soccer":
            ball_traces = input[1].unsqueeze(0).to(device)  # [1, time, 2]

        output_dim = model.params["n_features"]
        output_dim *= model.params["n_players"]
        if dataset in ["soccer", "basketball"]:
            output_dim *= 2

        seq_len = input_traces.shape[1]

        # Init episode ret
        episode_ret = {key: 0 for key in ret_keys}

        episode_pred = torch.zeros(seq_len, output_dim)
        episode_target = torch.zeros(seq_len, output_dim)
        episode_mask = torch.ones(seq_len, output_dim) * -1
        if model_name == "dbhp" and model.params["train_hybrid"]:
            if dataset == "afootball":
                episode_weights = torch.zeros(seq_len, model.params["n_players"] * 3)
            else:
                episode_weights = torch.zeros(seq_len, (model.params["n_players"] * 2) * 3)

        episode_pred_dict = {}
        output_dim = model.params["n_players"] * 2
        if dataset in ["soccer", "basketball"]:
            output_dim *= 2
        for key in model_keys:
            episode_pred_dict[key] = torch.zeros(seq_len, output_dim)  # [time, out_dim]

        for i in range(input_traces.shape[1] // wlen + 1):
            i_from = wlen * i
            i_to = wlen * (i + 1)

            window_input = input_traces[:, i_from:i_to]
            window_target = target_traces[:, i_from:i_to]
            if dataset == "soccer":  # For simulated camera view
                window_ball = ball_traces[:, i_from:i_to]

            window_target_ = reshape_tensor(window_target, dataset=dataset).flatten(2, 3)
            if window_input.shape[1] != wlen:
                episode_target[i_from:i_to] = window_target
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
            elif model.params["model"] == "graph_imputer":
                window_ret = model.evaluate(window_inputs, device=device)
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
                    calc_statistic_metrics(traces, targets, masks, window_ret, imputer=key, dataset=dataset)

            # Save sequence results
            episode_target[i_from:i_to] = window_ret["target"].detach().cpu().squeeze(0)
            episode_mask[i_from:i_to] = window_ret["mask"].detach().cpu().squeeze(0)
            for key in model_keys:
                if key == "pred":
                    episode_pred[i_from:i_to] = window_ret["pred"].detach().cpu().squeeze(0)
                else:
                    episode_pred_dict[key][i_from:i_to] = window_ret[f"{key}_pred"].detach().cpu().squeeze(0)

            for key in ret_keys:
                if key == "n_frames":
                    episode_ret[key] += seq_len
                elif key == "n_missings":
                    episode_ret[key] += ((1 - masks).sum() / model.params["n_features"]).item()
                else:
                    episode_ret[key] += window_ret[key]

            if model_name == "dbhp" and model.params["train_hybrid"]:
                episode_weights[i_from:i_to] = window_ret["train_hybrid_weights"].detach().cpu().squeeze(0)

        # Update episode ret
        episode_df_ret = {"target_df": episode_target, "mask_df": episode_mask}
        for key in model_keys + ["target"]:
            if key in ["pred", "target"]:
                episode_traces = episode_pred if key == "pred" else episode_target
                if model.params["normalize"]:
                    episode_traces = normalize_tensor(episode_traces, mode="upscale", dataset=dataset)
                episode_df_ret[f"{key}_df"] = episode_traces
            else:
                if model.params["normalize"]:
                    episode_pred_dict[key] = normalize_tensor(episode_pred_dict[key], mode="upscale", dataset=dataset)
                episode_df_ret[f"{key}_df"] = episode_pred_dict[key]

        if model_name == "dbhp" and model.params["train_hybrid"]:
            episode_df_ret["train_hybrid_weights"] = episode_weights
        return episode_ret, episode_df_ret

    def predict(self, model: nn.Module, statistic_metrics=False, gap_models=None, dataset="soccer"):
        model_name = model.params["model"]
        random.seed(1000)

        if model.params["n_features"] == 2:
            feature_types = ["_x", "_y"]
        elif model.params["n_features"] == 6:
            feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]

        players = self.team1_players + self.team2_players
        player_cols = [f"{p}{f}" for p in players for f in feature_types]

        model_keys = ["pred"]
        ret_keys = ["n_frames", "n_missings"]
        if model_name == "dbhp":
            if model.params["physics_loss"]:
                model_keys += ["physics_f", "physics_b"]
            if model.params["train_hybrid"]:
                model_keys += ["static_hybrid", "static_hybrid2", "train_hybrid"]
        if statistic_metrics:
            model_keys += ["linear", "knn", "forward"]

            metrics = ["speed", "change_of_step_size", "path_length"]
            ret_keys += [f"{m}_{metric}" for m in model_keys for metric in metrics]

        ret_keys += [f"{m}_dist" for m in model_keys]
        ret = {key: 0 for key in ret_keys}

        # Init results dataframes (predictions, mask)
        df_dict = TraceHelper.init_results_df(self.traces, model_keys, player_cols)
        if model_name == "dbhp" and model.params["train_hybrid"]:
            weights_cols = [f"{p}{w}" for p in players for w in ["_w0", "_w1", "_w2"]]
            hybrid_weight_df = self.traces.copy(deep=True)
            hybrid_weight_df[weights_cols] = -1

        x_cols = [c for c in self.traces.columns if c.endswith("_x")]
        y_cols = [c for c in self.traces.columns if c.endswith("_y")]

        if model.params["normalize"]:
            self.traces[x_cols] /= self.ps[0]
            self.traces[y_cols] /= self.ps[1]
            self.ps = (1, 1)

        for phase in self.traces["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase]
            phase_gks = SportsDataset.detect_goalkeepers(phase_traces)

            team1_code, team2_code = phase_gks[0][0], phase_gks[1][0]

            input_cols = [c for c in phase_traces[player_cols].dropna(axis=1).columns]
            team1_cols = [c for c in input_cols if c.startswith(team1_code)]
            team2_cols = [c for c in input_cols if c.startswith(team2_code)]
            ball_cols = [f"ball{t}" for t in ["_x", "_y"]]
            # Reorder teams so that the left team comes first.
            input_cols = team1_cols + team2_cols
            input_xy_cols = [c for c in input_cols if c.endswith("_x") or c.endswith("_y")]

            if min(len(team1_cols), len(team2_cols)) < model.params["n_features"] * model.params["n_players"]:
                continue

            episodes = [e for e in phase_traces["episode"].unique() if e > 0]
            for episode in tqdm(episodes, desc=f"Phase {phase}"):
                episode_traces = phase_traces[phase_traces["episode"] == episode]
                if len(episode_traces) < self.ws:
                    continue
                episode_player_traces = torch.FloatTensor(episode_traces[input_cols].values)
                episode_ball_traces = torch.FloatTensor(episode_traces[ball_cols].values)

                episode_input = [episode_player_traces, episode_ball_traces]

                with torch.no_grad():
                    episode_ret, episode_df_ret = TraceHelper.predict_episode(
                        episode_input,
                        ret_keys,
                        model_keys,
                        model,
                        statistic_metrics=statistic_metrics,
                        gap_models=gap_models,
                        dataset=dataset,
                    )

                # Update results dataframes (episode_predictions, episode_mask)
                TraceHelper.update_results_df(df_dict, episode_traces.index, input_cols, input_xy_cols, episode_df_ret)
                if model_name == "dbhp" and model.params["train_hybrid"]:
                    weight_player_cols = [c.split("_x")[0] for c in input_cols if "_x" in c]
                    input_weight_cols = [f"{p}{w}" for p in weight_player_cols for w in ["_w0", "_w1", "_w2"]]
                    hybrid_weight_df.loc[episode_traces.index, input_weight_cols] = np.array(
                        episode_df_ret["train_hybrid_weights"]
                    )

                for key in episode_ret:
                    ret[key] += episode_ret[key]

        if model.params["normalize"]:
            self.ps = (108, 72)
            self.traces[x_cols] *= self.ps[0]
            self.traces[y_cols] *= self.ps[1]

        if model_name == "dbhp" and model.params["train_hybrid"]:
            df_dict["train_hybrid_weights_df"] = hybrid_weight_df

        return ret, df_dict

    @staticmethod
    def init_results_df(traces, df_keys, player_cols):
        df_dict = {}
        df_dict["target_df"] = traces.copy(deep=True)
        df_dict["mask_df"] = traces.copy(deep=True)
        df_dict["mask_df"].loc[traces.index, player_cols] = -1

        for key in df_keys:
            df_dict[f"{key}_df"] = traces.copy(deep=True)

        return df_dict

    @staticmethod
    def update_results_df(df_dict, epi_idx, cols, xy_cols, epi_ret):
        for key in df_dict.keys():
            if key in ["pred_df", "target_df", "mask_df"]:
                df_dict[key].loc[epi_idx, cols] = np.array(epi_ret[key])
            else:
                df_dict[key].loc[epi_idx, xy_cols] = np.array(epi_ret[key])

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
