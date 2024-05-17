###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
import numpy as np
#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

# import models.baselines.latentode.latentode_utils as utils
# from models.baselines.latentode.latentode_utils import get_device
# from models.baselines.latentode.encoder_decoder import *
# from models.baselines.latentode.likelihood_eval import *

import models.latentode.latentode_utils as utils
from models.latentode.latentode_utils import get_device
from models.latentode.encoder_decoder import *
from models.latentode.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
# from models.baselines.latentode.base_models import VAE_Baseline
from models.latentode.base_models import VAE_Baseline

from models.utils import *

class LatentODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
		z0_prior, device, obsrv_std = None, 
		use_binary_classif = False, use_poisson_proc = False,
		linear_classifier = False,
		classif_per_tp = False,
		n_labels = 1,
		train_classif_w_reconstr = False):

		# super(LatentODE, self).__init__(
		super().__init__(
			input_dim = input_dim, latent_dim = latent_dim, 
			z0_prior = z0_prior, 
			device = device, obsrv_std = obsrv_std, 
			use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp, 
			linear_classifier = linear_classifier,
			use_poisson_proc = use_poisson_proc,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.use_poisson_proc = use_poisson_proc

	# def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
	# 	mask = None, n_traj_samples = 1, run_backwards = True, mode = None):
	def get_reconstruction(self, masked_x, mask):
		if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
			isinstance(self.encoder_z0, Encoder_z0_RNN):
			truth_w_mask = masked_x

			if mask is not None:
				truth_w_mask = torch.cat((masked_x, mask), -1)
			# first_point_mu, first_point_std = self.encoder_z0(
			# 	truth_w_mask, truth_time_steps, run_backwards = run_backwards)
			#MH Revised
			truth_time_steps = torch.linspace(0, 1, masked_x.shape[1]).to(self.device)
			time_steps_to_predict = torch.linspace(0, 1, masked_x.shape[1]).to(self.device)
			#MH Revised
			first_point_mu, first_point_std = self.encoder_z0(
				truth_w_mask, truth_time_steps, run_backwards = False)
			#MH Revised
			n_traj_samples = 1
			#MH Revised
			means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
			first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

		else:
			raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
		
		first_point_std = first_point_std.abs()
		assert(torch.sum(first_point_std < 0) == 0.)

		# if self.use_poisson_proc:
		# 	n_traj_samples, n_traj, n_dims = first_point_enc.size()
		# 	# append a vector of zeros to compute the integral of lambda
		# 	zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
		# 	first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
		# 	means_z0_aug = torch.cat((means_z0, zeros), -1)
		# else:
		first_point_enc_aug = first_point_enc
		means_z0_aug = means_z0
		
		assert(not torch.isnan(time_steps_to_predict).any())
		assert(not torch.isnan(first_point_enc).any())
		assert(not torch.isnan(first_point_enc_aug).any())

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

		# if self.use_poisson_proc:
		# 	sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

		# 	assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
		# 	assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = self.decoder(sol_y)

		all_extra_info = {
			"first_point": (first_point_mu, first_point_std, first_point_enc),
			"latent_traj": sol_y.detach()
		}

		# if self.use_poisson_proc:
		# 	# intergral of lambda from the last step of ODE Solver
		# 	all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
		# 	all_extra_info["log_lambda_y"] = log_lambda_y

		# if self.use_binary_classif:
		# 	if self.classif_per_tp:
		# 		all_extra_info["label_predictions"] = self.classifier(sol_y)
		# 	else:
		# 		all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

		return pred_x, all_extra_info


	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
		# input_dim = starting_point.size()[-1]
		# starting_point = starting_point.view(1,1,input_dim)

		# Sample z0 from prior
		starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

		starting_point_enc_aug = starting_point_enc
		if self.use_poisson_proc:
			n_traj_samples, n_traj, n_dims = starting_point_enc.size()
			# append a vector of zeros to compute the integral of lambda
			zeros = torch.zeros(n_traj_samples, n_traj,self.input_dim).to(self.device)
			starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

		sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
			n_traj_samples = 3)

		if self.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
		
		return self.decoder(sol_y)
	
	def forward(self, data, mode = 'train', device = 'cuda:0'):
		# if len(data) == 3: #Soccer 
		# 	x, y, ball = data
		# else:
		# 	x, y = data
		# 	ball = []
		if len(data) == 2: #Soccer 
			x, ball = data
		else:
			x = data[0]
			ball = []


		input_dict = {"target" : x, "ball" : ball}
		ret = dict()
		bs, seq_len, feat_dim = x.shape
		x = x.to(self.device)
		
		missing_probs = np.arange(10) * 0.1       
		# mask = generate_mask(
		# 	inputs = input_dict,
		# 	mode = self.missing_mode, 
		# 	ws = seq_len, 
		# 	missing_rate = missing_probs[random.randint(1, 9)],
		# 	dataset= self.dataset)
		mask, missing_rate = generate_mask(
			data = input_dict,
			sports = self.dataset,
			mode = self.missing_mode, 
			missing_rate = self.params['missing_rate']
			)
		

		# if self.missing_mode == 'camera':
		# 	time_gap = time_interval(mask, list(range(seq_len)), mode = 'camera')
		# 	mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # [1, time, n_players]
		# 	time_gap = torch.tensor(time_gap, dtype = torch.float32)
		# 	mask = torch.repeat_interleave(mask, self.n_features, dim=-1).squeeze(0)  # [bs, time, x_dim]
		# 	time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).squeeze(0)

		# else:
		# 	time_gap = time_interval(mask, list(range(seq_len)), mode = 'block')
		# 	mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # [1, time, n_players]
		# 	time_gap = torch.tensor(time_gap, dtype = torch.float32).unsqueeze(0)
		# 	mask = torch.repeat_interleave(mask, self.n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
		# 	time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
		
		time_gap = time_interval(mask, list(range(seq_len)), mode = 'camera')
		mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # [1, time, n_players]
		time_gap = torch.tensor(time_gap, dtype = torch.float32)
		mask = torch.repeat_interleave(mask, self.n_features, dim=-1).squeeze(0)  # [bs, time, x_dim]
		time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).squeeze(0)

		mask = mask.to(self.device)
		masked_x = x * mask

		pred_y, info = self.get_reconstruction(masked_x, mask)
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)

		assert(torch.sum(fp_std < 0) == 0.)

		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

		if torch.isnan(kldiv_z0).any():
			print(fp_mu)
			print(fp_std)
			raise Exception("kldiv_z0 is Nan!")


		# Mean over number of latent dimensions
		# kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
		# kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
		# shape after: [n_traj_samples]
		kldiv_z0 = torch.mean(kldiv_z0,(1,2))

		# Compute likelihood of all the points
		rec_likelihood = self.get_gaussian_likelihood(
			masked_x, pred_y,
			mask = mask)

		mse = self.get_mse(
			masked_x, pred_y,
			mask = mask)

		pois_log_likelihood = torch.Tensor([0.]).to(self.device)
		# if self.use_poisson_proc:
		# 	pois_log_likelihood = compute_poisson_proc_likelihood(
		# 		batch_dict["data_to_predict"], pred_y, 
		# 		info, mask = batch_dict["mask_predicted_data"])
		# 	# Take mean over n_traj
		# 	pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

		################################
		# Compute CE loss for binary classification on Physionet
		# device = get_device(batch_dict["data_to_predict"])
		# ce_loss = torch.Tensor([0.]).to(device)
		# if (batch_dict["labels"] is not None) and self.use_binary_classif:

		# 	if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
		# 		ce_loss = compute_binary_CE_loss(
		# 			info["label_predictions"], 
		# 			batch_dict["labels"])
		# 	else:
		# 		ce_loss = compute_multiclass_CE_loss(
		# 			info["label_predictions"], 
		# 			batch_dict["labels"],
		# 			mask = batch_dict["mask_predicted_data"])

		# IWAE loss
		kl_coef = 0.15
		loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)
			
		# if self.use_poisson_proc:
		# 	loss = loss - 0.1 * pois_log_likelihood 

		# if self.use_binary_classif:
		# 	if self.train_classif_w_reconstr:
		# 		loss = loss +  ce_loss * 100
		# 	else:
		# 		loss =  ce_loss

		updated_x = pred_y.squeeze(dim=0).clone()
		mask_idx = masked_x != 0
		updated_x[mask_idx] = masked_x[mask_idx]

		# pred_t = reshape_tensor(updated_x, rescale=True, dataset=self.dataset) # [bs, total_players, 2]
		# target_t = reshape_tensor(x, rescale=True, dataset=self.dataset)
		pred_t = reshape_tensor(updated_x, upscale=True, dataset_type=self.dataset) # [bs, total_players, 2]
		target_t = reshape_tensor(x, upscale=True, dataset_type=self.dataset)

		ret['target'] = x
        # ret['pred'] = pred_x
		ret['pred'] = updated_x
		ret['input'] = masked_x
		ret['mask'] = mask
		ret['total_loss'] = loss
		aggfunc = "mean" if mode=="train" else "sum"
		# ret['pred_dist'] = calc_trace_dist(ret["pred"], ret["target"], ret["mask"], n_features = self.n_features, aggfunc= aggfunc, dataset=self.dataset)
		ret['pred_dist'] = calc_pos_error(ret["pred"], ret["target"], ret["mask"], n_features = self.n_features, aggfunc= aggfunc, dataset=self.dataset)
		# ret['pred_dist'] = calc_trace_dist(ret["pred"], ret["target"], ret["mask"], n_features = self.n_features, aggfunc='sum', dataset=self.dataset)
	 
		return ret		


