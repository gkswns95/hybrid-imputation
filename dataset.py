import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SportsDataset(Dataset):
    def __init__(
        self,
        sports="soccer",  # ["soccer", "basketball", "afootball"]
        data_paths=None,
        n_features=6,
        window_size=200,
        episode_min_len=100,
        stride=5,
        normalize=False,
        flip_pitch=False,
    ):
        random.seed(1000)

        self.sports = sports
        if sports == "soccer":
            self.ps = (108, 72)
            self.team_size = 11  # number of input players per team
        elif sports == "bball":
            self.ps = (28.65, 15.24)
            self.team_size = 5  # number of input players per team
        else:  # sports == "afootball"
            self.ps = (110, 49)
            self.team_size = 6  # number of input players per team
            window_size = 50
            episode_min_len = 50

        self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"][:n_features]
        self.n_features = n_features
        self.ws = window_size
        self.flip_pitch = flip_pitch

        assert data_paths is not None

        halfline_x = 0.5 if normalize else self.ps[0] / 2
        pad_value = -100

        player_data_list = []
        ball_data_list = []

        for f in tqdm(data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            match_traces = pd.read_csv(f, header=0)

            if normalize:
                x_cols = [c for c in match_traces.columns if c.endswith("_x")]
                y_cols = [c for c in match_traces.columns if c.endswith("_y")]
                match_traces[x_cols] /= self.ps[0]
                match_traces[y_cols] /= self.ps[1]

            if sports == "soccer":
                player_cols = [c for c in match_traces.columns if c[0] in ["A", "B"] and c[3:] in self.feature_types]

                for phase in match_traces["phase"].unique():
                    phase_traces = match_traces[match_traces["phase"] == phase]

                    team1_gk, team2_gk = SportsDataset.detect_goalkeepers(phase_traces, halfline_x)
                    team1_code, team2_code = team1_gk[0], team2_gk[0]

                    input_cols = phase_traces[player_cols].dropna(axis=1).columns
                    team1_cols = [c for c in input_cols if c[0] == team1_code]
                    team2_cols = [c for c in input_cols if c[0] == team2_code]
                    if min(len(team1_cols), len(team2_cols)) < n_features * self.team_size:
                        continue

                    # Reorder teams so that the left team comes first
                    input_cols = team1_cols + team2_cols

                    episodes = [e for e in phase_traces["episode"].unique() if e > 0]
                    for ep in episodes:
                        ep_traces = match_traces[match_traces["episode"] == ep]
                        ep_len = len(ep_traces)

                        ep_player_data = ep_traces[input_cols].values
                        ep_ball_data = ep_traces[["ball_x", "ball_y"]].values

                        if ep_len > episode_min_len and ep_len < self.ws:
                            pad_len = self.ws - ep_len
                            ep_player_data = np.pad(ep_player_data, ((0, pad_len), (0, 0)), constant_values=pad_value)
                            ep_ball_data = np.pad(ep_ball_data, ((0, pad_len), (0, 0)), constant_values=pad_value)
                            player_data_list.append(ep_player_data)
                            ball_data_list.append(ep_ball_data)

                        elif ep_len >= self.ws:
                            for i in range(ep_len - self.ws + 1):
                                if i % stride == 0 or i == ep_len - self.ws:
                                    player_data_list.append(ep_player_data[i : i + self.ws])
                                    ball_data_list.append(ep_ball_data[i : i + self.ws])

            else:
                if sports == "basketball":
                    player_cols = [f"player{i}{x}" for i in range(self.team_size * 2) for x in self.feature_types]
                else:  # if sports == "afootball"
                    player_cols = [f"player{i}{x}" for i in range(self.team_size) for x in self.feature_types]

                episodes = [e for e in match_traces["episode"].unique() if e > 0]
                for ep in episodes:
                    ep_player_data = match_traces[match_traces["episode"] == ep][player_cols]
                    ep_len = len(ep_player_data)

                    if ep_len > episode_min_len and ep_len < self.ws:
                        pad_len = self.ws - ep_len
                        ep_player_data = np.pad(ep_player_data, ((0, pad_len), (0, 0)), constant_values=pad_value)
                        player_data_list.append(ep_player_data)

                    elif ep_len >= self.ws:
                        for i in range(ep_len - self.ws + 1):
                            if i % stride == 0 or i == ep_len - self.ws:
                                player_data_list.append(ep_player_data[i : i + self.ws])

        player_data = np.stack(player_data_list, axis=0) if player_data_list else np.array([])
        ball_data = np.stack(ball_data_list, axis=0) if ball_data_list else np.array([])  # only for soccer dataset

        if normalize:
            self.ps = (1, 1)

        if n_features < 6:
            player_data = player_data.reshape(player_data.shape[0], self.ws, -1, len(self.feature_types))
            player_data = player_data[..., :n_features].reshape(player_data.shape[0], self.ws, -1)

        self.player_data = torch.FloatTensor(player_data)
        self.ball_data = torch.FloatTensor(ball_data)  # only for soccer dataset

    def __getitem__(self, i):
        if self.sports != "basketball" and self.flip_pitch:
            player_data = np.copy(self.player_data[i])

            flip_x = np.random.choice(2, (1, 1))
            flip_y = np.random.choice(2, (1, 1))

            # (ref, mul) = (ps, -1) if flip == 1 else (0, 1)
            ref_x = flip_x * self.ps[0]
            ref_y = flip_y * self.ps[1]
            mul_x = 1 - flip_x * 2
            mul_y = 1 - flip_y * 2

            # flip (x, y) locations
            player_data[:, 0 :: self.n_features] = player_data[:, 0 :: self.n_features] * mul_x + ref_x
            player_data[:, 1 :: self.n_features] = player_data[:, 1 :: self.n_features] * mul_y + ref_y

            # flip (x, y) velocity and acceleration values
            if self.n_features > 2:
                player_data[:, 2 :: self.n_features] = player_data[:, 2 :: self.n_features] * mul_x
                player_data[:, 3 :: self.n_features] = player_data[:, 3 :: self.n_features] * mul_y
                player_data[:, 4 :: self.n_features] = player_data[:, 4 :: self.n_features] * mul_x
                player_data[:, 5 :: self.n_features] = player_data[:, 5 :: self.n_features] * mul_y

            # if flip_x == 1, reorder team1 and team2 features
            team1_input = player_data[:, : self.n_features * self.team_size]
            team2_input = player_data[:, self.n_features * self.team_size :]

            input_permuted = np.concatenate([team2_input, team1_input], -1)
            player_data = torch.FloatTensor(np.where(flip_x, input_permuted, player_data))

            if self.sports == "soccer":
                ball_data = np.copy(self.ball_data[i])
                ball_data[:, [0]] = ball_data[:, [0]] * mul_x + ref_x
                ball_data[:, [1]] = ball_data[:, [1]] * mul_y + ref_y
                return player_data, ball_data
            else:
                return player_data

        else:
            return self.player_data[i], self.ball_data[i] if self.sports == "soccer" else self.player_data[i]

    def __len__(self):
        return len(self.player_data)

    @staticmethod
    def detect_goalkeepers(traces: pd.DataFrame, halfline_x=54):
        a_x_cols = [c for c in traces.columns if c.startswith("A") and c.endswith("_x")]
        b_x_cols = [c for c in traces.columns if c.startswith("B") and c.endswith("_x")]

        a_gk = (traces[a_x_cols].mean() - halfline_x).abs().idxmax()[:3]
        b_gk = (traces[b_x_cols].mean() - halfline_x).abs().idxmax()[:3]

        a_gk_mean_x = traces[f"{a_gk}_x"].mean()
        b_gk_mean_y = traces[f"{b_gk}_x"].mean()

        return (a_gk, b_gk) if a_gk_mean_x < b_gk_mean_y else (b_gk, a_gk)
