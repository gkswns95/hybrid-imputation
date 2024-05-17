import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint

from models.utils import *

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out
def log_normal_pdf(x, mean, logvar):
    # const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.default_device)
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class LatentODE(nn.Module):
    def __init__(self, params, parser = None):
        super().__init__()
        self.model_args = [
            "n_players",
            "pe_z_dim",
            "pi_z_dim",
            "rnn_dim",
            "hybrid_rnn_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "pred_xy",
            "physics_loss",
            "cartesian_accel",
            "transformer",
            "ppe",
            "fpe",
            "fpi",
            "train_hybrid",
            "bidirectional",
            "dynamic_missing",
            "upper_bound",
            "stochastic",
            "seperate_learning",
            "avg_length_loss",
            "var_dim",
            "kld_weight_float",
            "weighted",
            "missing_prob_float",
            "m_pattern"
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)
        self.device = self.params['device']
        self.missing_mode = "block" if self.params["dynamic_missing"] else "block_all_feat"
        if self.params["dynamic_missing"]:
            if self.params["m_pattern"] == "camera_simulate":
                self.missing_mode = "camera_simulate"
            else:
                self.missing_mode = "player_wise"
        else:
            self.missing_mode = "all_player"

        self.dataset = params['dataset']
        self.n_features = params['n_features']
        self.latent_dim = params['var_dim']
        self.nhidden = params['rnn_dim']
        self.rnn_nhidden = params['rnn_dim']
        self.obs_dim = params['n_players'] * params['n_features'] * 2
        self.nbatch = params['batch_size']

        self.func = LatentODEfunc(self.latent_dim, self.nhidden)
        self.rec = RecognitionRNN(self.latent_dim, self.obs_dim, self.rnn_nhidden, self.nbatch)
        self.dec = Decoder(self.latent_dim, self.obs_dim, self.nhidden)

        self.ts = np.linspace(0, 1, num = 100)
        self.ts = torch.from_numpy(self.ts).float().to(self.device)
    
    def forward(self, data, device = 'cuda:0'):
        if len(data) == 3: #Soccer
            x, y, ball = data
        else:
            x, y = data
            ball = []
    
        input_dict = {"target" : x, "ball" : ball}
        ret = dict()
        bs, seq_len, feat_dim = x.shape
        x = x.to(self.device)
        
        missing_probs = np.arange(10) * 0.1       
        mask = generate_mask(
            inputs = input_dict,
            mode = self.missing_mode, 
            ws = seq_len, 
            missing_rate = missing_probs[random.randint(1, 9)],
            dataset= self.dataset)
        
        if self.missing_mode == 'camera_simulate':
            time_gap = time_interval(mask, list(range(seq_len)), mode = 'camera')
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # [1, time, n_players]
            time_gap = torch.tensor(time_gap, dtype = torch.float32)
            mask = torch.repeat_interleave(mask, self.n_features, dim=-1).squeeze(0)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).squeeze(0)

        else:
            time_gap = time_interval(mask, list(range(seq_len)), mode = 'block')
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # [1, time, n_players]
            time_gap = torch.tensor(time_gap, dtype = torch.float32).unsqueeze(0)
            mask = torch.repeat_interleave(mask, self.n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
        
        mask = mask.to(self.device)
        masked_x = x * mask

        #Forward
        self.rec.nbatch = bs
        h = self.rec.initHidden().to(self.device)
        for t in reversed(range(masked_x.size(1))):
            obs = masked_x[:, t, :]
            out, h = self.rec.forward(obs, h)
    
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(self.device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, self.ts).permute(1, 0, 2)
        pred_x = self.dec(pred_z)

        updated_x = pred_x.clone()
        mask_idx = masked_x != 0
        updated_x[mask_idx] = masked_x[mask_idx]

        # compute loss
        noise_std = .3
        noise_std_ = torch.zeros(pred_x.size()).to(self.device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(self.device)
        logpx = log_normal_pdf(
            x, updated_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(self.device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        
        pred_t = reshape_tensor(updated_x, rescale=True, dataset=self.dataset) # [bs, total_players, 2]
        target_t = reshape_tensor(x, rescale=True, dataset=self.dataset)
        
        ret['target'] = x
        # ret['pred'] = pred_x
        ret['pred'] = updated_x
        ret['mask'] = mask
        ret['total_loss'] = loss
        ret['pred_dist'] = calc_trace_dist(ret["pred"], ret["target"], ret["mask"], n_features = self.n_features, aggfunc='mean', dataset=self.dataset)
        # ret['pred_dist'] = calc_trace_dist(ret["pred"], ret["target"], ret["mask"], n_features = self.n_features, aggfunc='sum', dataset=self.dataset)
        
        return ret



                