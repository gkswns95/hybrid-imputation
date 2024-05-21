import os
import sys
from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import datatools.matplotsoccer as mps
from models.utils import calc_pos_error, get_dataset_config, reshape_tensor

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())


class VisualizeHelper:
    def __init__(
        self,
        trial: str,
        df_dict: Dict[str, pd.DataFrame],
        plot_mode: str,
        dataset: str,
        helper,
    ):
        self.trial = trial
        self.traces = helper.traces

        self.mode = plot_mode
        self.dataset = dataset
        self.df_dict = df_dict

        self.wlen = 200

        if dataset == "soccer":
            players = helper.team1_players + helper.team2_players
        elif dataset == "basketball":
            players = ["player" + str(i) for i in range(helper.total_players)]

        self.p_cols = [f"{p}{f}" for p in players for f in ["_x", "_y"]]
        if "hybrid_d" in df_dict.keys():
            self.w_cols = [f"{p}{w}" for p in players for w in ["_w0", "_w1", "_w2"]]

    def valid_episodes(self):
        episodes = self.traces["episode"].unique()

        valid_episodes = []
        for e in episodes:
            traces = self.traces[self.traces["episode"] == e]
            if traces.shape[0] >= self.wlen and e != 0:
                valid_episodes.append(e)

        self.val_epi_list = np.array(valid_episodes)
        print(f"Valid episode idxs : {valid_episodes}")

    def get_cfg(self):
        pred_keys = {}
        fs = (16, 3)
        if self.mode == "imputed_traj":
            fs = (16, 5) # width, height
            pred_keys.update(
                {
                    "mask": "mask",
                    "target": "Ground Truth",
                    "hybrid_d": "DBHP-D",
                    "knn": "k-NN",
                    "linear": "Linear Interpolation",
                    "linear_4": "BRITS",
                    "linear_3": "NAOMI",
                    "linear_1": "NRTSI",
                    "linear_2": "Graph Imputer",
                }
            )
        elif self.mode == "dist_heatmap":
            pred_keys.update(
                {
                    "pred": "",
                    "dap_f": "",
                    "dap_b": "",
                    "hybrid_s2": "",
                    "hybrid_d": "",
                }
            )
        elif self.mode == "weights_heatmap":
            pred_keys["lambdas"] = "lambdas"

        pred_keys.update({"target": "Ground Truth", "mask": "mask"})

        return pred_keys, fs

    def plot_trajectories(self):
        n_players, ps = get_dataset_config(self.dataset)
        for i, (key, title) in enumerate(self.pred_keys.items()):
            if key == "mask":
                continue

            # For debugging
            if key.startswith("tmp"):
                key = "linear"
            ax = self.fig.add_subplot(4, 2, i)
            ax.set_xlim(0, ps[0])
            ax.set_ylim(0, ps[1])

            if self.dataset == "soccer":
                mps.field("blue", self.fig, ax, show=False)
                s1 = 0.8
                s2 = 3
                lw = 0.5
                c = [
                    "#7D3C98",
                    "#00A591",
                    "#3B3B98",
                    "#E89B98",
                    "#FAB5DA",
                    "#FF5733",
                    "#33FF57",
                    "#5733FF",
                    "#FFFF33",
                    "#33FFFF",
                    "#FF33FF",
                ]
                # plt.subplots_adjust(wspace=0.1)
            elif self.dataset == "basketball":
                court = plt.imread("img/basketball_court.png")
                s1 = 1
                s2 = 2
                lw = 1
                # s1 = 10
                # s2 = 20
                # lw = 1.5
                c = ["#7D3C98", "#00A591", "#3B3B98", "#E89B98", "#FAB5DA"]
                ax.imshow(court, zorder=0, extent=[0, ps[0], ps[1], 0])

            w_traces = self.pred_dict[f"window_{key}"]
            w_mask = self.pred_dict["window_mask"]
            for i, k in enumerate(range(n_players // 2)):
                obs_idxs = np.where((w_mask[:, 2 * k] != 0) | (w_mask[:, 2 * k + 1] != 0))
                missing_idxs = np.setdiff1d(np.arange(w_traces.shape[0]), obs_idxs)

                ax.scatter(
                    w_traces[missing_idxs, 2 * k],
                    w_traces[missing_idxs, 2 * k + 1],
                    marker="o",
                    color=c[i],
                    s=s1,
                    alpha=0.6,
                    zorder=2,
                )
                ax.plot(w_traces[:, 2 * k], w_traces[:, 2 * k + 1], color=c[i], linewidth=lw, alpha=1, zorder=2)
                ax.scatter(
                    w_traces[0, 2 * k], w_traces[0, 2 * k + 1], marker="o", color="black", s=s2, alpha=1, zorder=4
                )  # starting point

                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False,
                )
                ax.set_title(title.format(i + 1), fontsize=10, loc="center")

    def plot_hybrid_weights(self):
        mask = self.pred_dict["window_mask"]
        lambdas = self.pred_dict["window_lambdas"]

        m = reshape_tensor(mask, dataset_type=self.dataset)
        m = m[..., 1].squeeze(-1)
        m = np.array((m == 1))

        for i, title in enumerate(["STRNN-DP", "STRNN-DAP-F", "STRNN-DAP-B"]):
            ax = self.fig.add_subplot(1, 4, i + 1)

            sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=True, mask=m, ax=ax, vmin=0, vmax=1)

            ax.set_xlabel("Agent", fontsize=12)
            ax.set_ylabel("Time step", fontsize=12)
            # ax.set_title(title.format(i + 1), fontsize=20, loc="center")

            ax.set_yticks([0, 50, 100, 150, 200])
            ax.set_yticklabels([0, 50, 100, 150, 200], rotation=0)

        return self.fig

    def plot_dist_heatmap(self):
        target = self.pred_dict["window_target"]
        mask = self.pred_dict["window_mask"]

        i = 0
        for key, title in self.pred_keys.items():
            if key in ["target", "mask"]:
                continue

            ax = self.fig.add_subplot(1, 6, i + 1)

            pred = self.pred_dict[f"window_{key}"]

            pred_dist = calc_pos_error(pred, target, mask, upscale=False, aggfunc="tensor", dataset=self.dataset)

            m = reshape_tensor(mask, dataset_type=self.dataset)
            m = m[..., 1].squeeze(-1)
            m = np.array((m == 1))

            sns.heatmap(pred_dist
                        , cmap="viridis", 
                        cbar=True, 
                        mask=m, 
                        ax=ax)

            ax.set_title(f"{title} L2 distance")
            ax.set_xlabel("Agents", fontsize=12)
            ax.set_ylabel("Timesteps", fontsize=12)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticks([0, 50, 100, 150, 200])
            ax.set_yticklabels([0, 50, 100, 150, 200], rotation=0)
            # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_title(title.format(i + 1), fontsize=18, loc="center", y=1.05)

            i += 1

    def plot_run(self, epi_idx):
        e = self.val_epi_list[epi_idx]
        path = f"plots/{self.dataset}/{self.trial}/{self.mode}/episode_{e}"
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Plotting episode : {e}")
        print(f"Plot Mode : {self.mode}")
        print(f"Saved path : {path}")

        epi_traces = self.traces[self.traces["episode"] == e]

        self.pred_keys, fs = self.get_cfg()
        self.pred_dict = {}
        for seq, i in enumerate(range(epi_traces.shape[0] // self.wlen + 1)):
            self.fig = plt.figure(figsize=fs, dpi=300)

            i_from = self.wlen * i
            i_to = self.wlen * (i + 1)

            if epi_traces[i_from:i_to].shape[0] != self.wlen:
                continue

            # Processing window inputs
            for key in self.pred_keys.keys():
                # For debugging
                if key.startswith("tmp"):
                    continue

                cols = self.w_cols if key == "lambdas" else self.p_cols
                if key.startswith("linear"):
                    epi_df = self.df_dict["linear"][self.df_dict["linear"]["episode"] == e][cols]    
                else:
                    epi_df = self.df_dict[key][self.df_dict[key]["episode"] == e][cols]
                # epi_df = self.df_dict[f"{key}_df"][self.df_dict[f"{key}_df"]["episode"] == e][cols]
                epi_df = epi_df[i_from:i_to].replace(-1, np.nan)
                self.pred_dict[f"window_{key}"] = torch.tensor(epi_df.dropna(axis=1).values)

            # Plotting start
            if self.mode == "imputed_traj":
                self.plot_trajectories()
            elif self.mode == "dist_heatmap":
                self.plot_dist_heatmap()
            elif self.mode == "weights_heatmap":
                self.plot_hybrid_weights()

            plt.tight_layout()
            # plt.subplots_adjust(wspace=-0.5, hspace=0.3)
            self.fig.savefig(f"{path}/seq_{seq}", bbox_inches="tight")
            plt.close()