import os

import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import Dataset
from tqdm import tqdm

class SoccerDataset(Dataset):
    def __init__(
        self,
        data_paths=None,
        target_type="imputation",
        train=True,
        load_saved=False,
        save_new=False,
        n_features=6,
        cartesian_accel=False,
        window_size=200,
        pitch_size=(108, 72),
        normalize=False,
        flip_pitch=False,
        overlap=True,
    ):  
        random.seed(1000)
        self.n_pass = 0
        self.n_pass_miss = 0
        self.missing_value = 0
        self.target_type = target_type
        self.cartesian_accel = cartesian_accel # shold be removed
        if n_features == 6:
            self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
            # if cartesian_accel:
            #     self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]  # total features to save as npy files
            # else:
            #     self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"]  # total features to save as npy files
        else:
            self.feature_types = ["_x", "_y"]
        # number of features among self.feature_types to use in model training
        self.n_features = n_features
        self.k = 11  # number of input players per each team

        self.ws = window_size
        self.n_sliding = 30 # number of sliding window
        self.ps = pitch_size
        self.augment = flip_pitch
        self.overlap = overlap

        if load_saved:
            load_dir = f"data/save_data/train_{train}_n_features_{n_features}"
            self.input_data = torch.load(load_dir + "_input.pkl")
            self.target_data = torch.load(load_dir + "_target.pkl")

            print(f"Load processed data : {load_dir}")

        else:
            ### debug ###
            # val_epi_count = 0
            ### debug ###
            assert data_paths is not None

            targets = [target_type]
            halfline_x = 0.5 if normalize else self.ps[0] / 2

            input_data_list = []
            target_data_list = []
            ball_data_list = []
            for f in tqdm(data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                match_traces = pd.read_csv(f, header=0)
                player_cols = [c for c in match_traces.columns if c[0] in ["A", "B"] and c[3:] in self.feature_types]

                if normalize:
                    x_cols = [c for c in match_traces.columns if c.endswith("_x")]
                    y_cols = [c for c in match_traces.columns if c.endswith("_y")]
                    match_traces[x_cols] /= self.ps[0]
                    match_traces[y_cols] /= self.ps[1]

                for phase in match_traces["phase"].unique():
                    if type(phase) == str:  # For GPS-event traces, ignore phases with n_players < 22
                        phase_tuple = [int(i) for i in phase[1:-1].split(",")]
                        if phase_tuple[0] < 0 or phase_tuple[1] < 0:
                            continue

                    phase_traces = match_traces[match_traces["phase"] == phase]

                    team1_gk, team2_gk = SoccerDataset.detect_goalkeepers(phase_traces, halfline_x)
                    team1_code, team2_code = team1_gk[0], team2_gk[0]

                    input_cols = [c for c in phase_traces[player_cols].dropna(axis=1).columns if c[:3] not in targets]
                    team1_cols = [c for c in input_cols if c.startswith(team1_code)]
                    team2_cols = [c for c in input_cols if c.startswith(team2_code)]
                    ball_cols  = [f"ball{t}" for t in ["_x", "_y"]]

                    # Reorder teams so that the left team comes first
                    input_cols = team1_cols + team2_cols
                    
                    if min(len(team1_cols), len(team2_cols)) < n_features * self.k:
                        continue
                    
                    episodes = [e for e in phase_traces["episode"].unique() if e > 0]
                    for episode in episodes:
                        episode_traces = match_traces[match_traces["episode"] == episode]
                        episode_input = episode_traces[input_cols].values
                        if target_type == "imputation":
                            episode_target = episode_traces[input_cols].values
                            episode_ball = episode_traces[ball_cols].values
                            if len(episode_traces) >= self.ws:
                                ### debug ###
                                val_epi_count += 1
                                ### debug ###
                                for i in range(0, len(episode_traces) - self.ws + 1, self.ws):
                                    if self.overlap:
                                        if i + self.n_sliding + self.ws < len(episode_traces):
                                            for j in range(self.n_sliding):
                                                input_data_list.append(episode_input[i + j: i + j + self.ws])
                                                target_data_list.append(episode_target[i + j: i + j + self.ws])
                                                ball_data_list.append(episode_ball[i + j: i + j + self.ws])
                                    else:
                                        input_data_list.append(episode_input[i: i + self.ws])
                                        target_data_list.append(episode_target[i: i + self.ws])
                                        ball_data_list.append(episode_ball[i: i + self.ws])

            input_data = np.stack(input_data_list, axis=0)
            target_data = np.stack(target_data_list, axis=0)
            ball_data = np.stack(ball_data_list, axis=0)

            # ### debug ###
            # print(f"val_epi_count : {val_epi_count}")
            # print(f"input_data : {input_data.shape}")
            # asd
            # ### debug ###

            if normalize:
                self.ps = (1, 1)

            if n_features < 6:
                input_data = input_data.reshape(input_data.shape[0], self.ws, -1, len(self.feature_types))
                input_data = input_data[:, :, :, :n_features].reshape(input_data.shape[0], self.ws, -1)

            self.input_data = torch.FloatTensor(input_data)
            
            if target_type == "imputation":
                self.target_data = torch.FloatTensor(target_data)
                self.ball_data = torch.FloatTensor(ball_data)

            if save_new:
                save_dir = f"data/save_data/train_{train}_n_features_{n_features}"
                torch.save(self.input_data, save_dir + "_input.pkl")
                torch.save(self.target_data, save_dir + "_target.pkl")

                print(f"Save processed data : {save_dir}")

    def __getitem__(self, i):
        if self.target_type == "imputation":
            if self.augment:
                input_data = np.copy(self.input_data[i])
                ball_data = np.copy(self.ball_data[i])

                flip_x = np.random.choice(2, (1, 1))
                flip_y = np.random.choice(2, (1, 1))
                # valid input dimension only including player features
                valid_dim = self.n_features * (self.k * 2)

                # (ref, mul) = (ps, -1) if flip == 1 else (0, 1)
                ref_x = flip_x * self.ps[0]
                ref_y = flip_y * self.ps[1]
                mul_x = 1 - flip_x * 2
                mul_y = 1 - flip_y * 2
                
                # flip x and y
                input_data[:, 0:valid_dim:self.n_features] = input_data[:, 0:valid_dim:self.n_features] * mul_x + ref_x
                input_data[:, 1:valid_dim:self.n_features] = input_data[:, 1:valid_dim:self.n_features] * mul_y + ref_y

                ball_data[:, [0]] = ball_data[:, [0]] * mul_x + ref_x
                ball_data[:, [1]] = ball_data[:, [1]] * mul_y + ref_y

                # flip vx,vy,ax,ay
                if self.n_features > 2:
                    input_data[:, 2:valid_dim:self.n_features] = input_data[:, 2:valid_dim:self.n_features] * mul_x
                    input_data[:, 3:valid_dim:self.n_features] = input_data[:, 3:valid_dim:self.n_features] * mul_y
                    if self.cartesian_accel:
                        input_data[:, 4:valid_dim:self.n_features] = input_data[:, 4:valid_dim:self.n_features] * mul_x
                        input_data[:, 5:valid_dim:self.n_features] = input_data[:, 5:valid_dim:self.n_features] * mul_y

                # if flip_x == 1, reorder team1 and team2 features
                team1_input = input_data[:, :self.n_features * self.k]
                team2_input = input_data[:, self.n_features * self.k: valid_dim]

                input_permuted = np.concatenate([team2_input, team1_input], -1)
                input_data = torch.FloatTensor(np.where(flip_x, input_permuted, input_data))
                
                target_data = input_data.clone()
                
                return input_data, target_data, ball_data
            else:
                return self.input_data[i], self.target_data[i], self.ball_data[i]
        else:
            return self.input_data[i], self.target_data[i], self.ball_data[i]

    def __len__(self):
        return len(self.input_data)

    @staticmethod
    def detect_goalkeepers(traces: pd.DataFrame, halfline_x=54):
        a_x_cols = [c for c in traces.columns if c.startswith("A") and c.endswith("_x")]
        b_x_cols = [c for c in traces.columns if c.startswith("B") and c.endswith("_x")]

        a_gk = (traces[a_x_cols].mean() - halfline_x).abs().idxmax()[:3]
        b_gk = (traces[b_x_cols].mean() - halfline_x).abs().idxmax()[:3]

        a_gk_mean_x = traces[f"{a_gk}_x"].mean()
        b_gk_mean_y = traces[f"{b_gk}_x"].mean()

        return (a_gk, b_gk) if a_gk_mean_x < b_gk_mean_y else (b_gk, a_gk)

class NBAdataset(Dataset):
    def __init__(
        self,
        data_paths=None,
        target_type="imputation",
        train=True,
        load_saved=False,
        save_new=False,
        n_features=6,
        cartesian_accel=False,
        window_size=200,
        pitch_size=(28.65, 15.24),
        normalize=False,
        flip_pitch=False,
        overlap=True,
    ):      
        self.target_type = target_type
        self.cartesian_accel = cartesian_accel
        if n_features == 6:
            self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
            # if cartesian_accel:
            #     self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]  # total features to save as npy files
            # else:
            #     self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"]  # total features to save as npy files
        else:
            self.feature_types = ["_x", "_y"]
        
        # number of features among self.feature_types to use in model training
        self.n_features = n_features
        self.k = 5  # number of input players per each team

        self.ws = window_size
        self.n_sliding = 10 # number of sliding window
        self.ps = pitch_size
        self.augment = flip_pitch
        self.overlap = overlap
        
        input_data_list = []
        target_data_list = []
        ### debug ###
        val_epi_count = 0
        ### debug ###
        for f in tqdm(data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            match_traces = pd.read_csv(f, header=0)
            input_cols = [f"player{c}{f}" for c in range(self.k * 2) for f in self.feature_types]
            
            if normalize:
                x_cols = [c for c in input_cols if c.endswith("_x")]
                y_cols = [c for c in input_cols if c.endswith("_y")]
                match_traces[x_cols] /= self.ps[0]
                match_traces[y_cols] /= self.ps[1]

            episodes = [e for e in match_traces["episode"].unique() if e > 0]
            for episode in episodes:
                episode_traces = match_traces[match_traces["episode"] == episode]
                episode_input = episode_traces[input_cols].values
                if target_type == "imputation":
                    episode_target = episode_traces[input_cols].values
                    if len(episode_traces) >= self.ws:
                        ### debug ###
                        val_epi_count += 1
                        ### debug ###
                        for i in range(0, len(episode_traces) - self.ws + 1, self.ws):
                            if overlap:
                                if i + self.n_sliding + self.ws < len(episode_traces):
                                    for j in range(self.n_sliding):
                                        input_data_list.append(episode_input[i + j: i + j + self.ws])
                                        target_data_list.append(episode_target[i + j: i + j + self.ws])
                            else:
                                input_data_list.append(episode_input[i: i + self.ws])
                                target_data_list.append(episode_target[i: i + self.ws])

            input_data = np.stack(input_data_list, axis=0)
            target_data = np.stack(target_data_list, axis=0)

        if n_features < 6:
            input_data = input_data.reshape(input_data.shape[0], self.ws, -1, len(self.feature_types))
            input_data = input_data[:, :, :, :n_features].reshape(input_data.shape[0], self.ws, -1)

        self.input_data = torch.FloatTensor(input_data)
        self.target_data = torch.FloatTensor(target_data)

        ### debug ###
        # print(f"val_epi_count : {val_epi_count}")
        # print(f"input_data : {input_data.shape}")
        # asd
        ### debug ###

    def __getitem__(self, i):
        if self.target_type == "imputation":
            return self.input_data[i], self.target_data[i]

    def __len__(self):
        return len(self.input_data)

class NFLdataset(Dataset):
    def __init__(
        self,
        data_paths=None,
        target_type="imputation",
        train=True,
        load_saved=False,
        save_new=False,
        n_features=6,
        cartesian_accel=False,
        window_size=50,
        pitch_size=(110, 49),
        normalize=False,
        flip_pitch=False,
        overlap=True,
    ):      
        self.target_type = target_type
        self.cartesian_accel = cartesian_accel
        if n_features == 6:
            self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
            # if cartesian_accel:
            #     self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]  # total features to save as npy files
            # else:
            #     self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"]  # total features to save as npy files
        else:
            self.feature_types = ["_x", "_y"]
        
        # number of features among self.feature_types to use in model training
        self.n_features = n_features
        self.k = 6  # number of input players per each team

        self.ws = window_size
        self.n_sliding = 10 # number of sliding window
        self.ps = pitch_size
        self.augment = flip_pitch
        self.overlap = overlap
        ### debug ###
        val_epi_count = 0 
        ### debug ###
        input_data_list = []
        target_data_list = []
        for f in tqdm(data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            match_traces = pd.read_csv(f, header=0)
            input_cols = [f"player{c}{f}" for c in range(self.k) for f in self.feature_types]
            
            if normalize:
                x_cols = [c for c in input_cols if c.endswith("_x")]
                y_cols = [c for c in input_cols if c.endswith("_y")]
                match_traces[x_cols] /= self.ps[0]
                match_traces[y_cols] /= self.ps[1]

            episodes = [e for e in match_traces["episode"].unique() if e > 0]
            for episode in episodes:
                ### debug ###
                val_epi_count += 1
                ### debug ###
                episode_traces = match_traces[match_traces["episode"] == episode]
                episode_input = episode_traces[input_cols].values
                if target_type == "imputation":
                    episode_target = episode_traces[input_cols].values
                    for i in range(0, len(episode_traces) - self.ws + 1, self.ws):
                        input_data_list.append(episode_input[i: i + self.ws])
                        target_data_list.append(episode_target[i: i + self.ws])

            input_data = np.stack(input_data_list, axis=0)
            target_data = np.stack(target_data_list, axis=0)

        if n_features < 6:
            input_data = input_data.reshape(input_data.shape[0], self.ws, -1, len(self.feature_types))
            input_data = input_data[:, :, :, :n_features].reshape(input_data.shape[0], self.ws, -1)

        self.input_data = torch.FloatTensor(input_data)
        self.target_data = torch.FloatTensor(target_data)

        ### debug ###
        # print(f"val_epi_count : {val_epi_count}")
        # print(f"input_data : {input_data.shape}")
        # asd
        ### debug ###

    def __getitem__(self, i):
        if self.target_type == "imputation":
            return self.input_data[i], self.target_data[i]

    def __len__(self):
        return len(self.input_data)

if __name__ == "__main__":
    dir = "data/metrica_traces"
    filepaths = [f"{dir}/{f}" for f in os.listdir(dir) if f.endswith(".csv")]
    filepaths.sort()
    dataset = SoccerDataset(filepaths[-1:], target_type="gk", train=False, save=False)
    print(dataset[10000][2])