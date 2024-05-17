###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.distributions.normal import Normal

# import models.baselines.latentode.latentode_utils as utils
# from models.baselines.latentode.latent_ode import LatentODE
# from models.baselines.latentode.encoder_decoder import *
# from models.baselines.latentode.diffeq_solver import DiffeqSolver
# from models.baselines.latentode.ode_func import ODEFunc, ODEFunc_w_Poisson
import models.latentode.latentode_utils as utils
from models.latentode.latent_ode import LatentODE
from models.latentode.encoder_decoder import *
from models.latentode.diffeq_solver import DiffeqSolver
from models.latentode.ode_func import ODEFunc, ODEFunc_w_Poisson


from models.utils import *

#####################################################################################################

# def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
# 	classif_per_tp = False, n_labels = 1):
def create_LatentODE_model(params, parser):
	model_args = [
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
	params = parse_model_params(model_args, params, parser)
	params_str = get_params_str(model_args, params)

	# dim = args.latents
	dim = params['var_dim']
	device = params['device']
	input_dim = params['n_players'] * params['n_features'] * 2
	# if args.poisson:
	# 	lambda_net = utils.create_net(dim, input_dim, 
	# 		n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

	# 	# ODE function produces the gradient for latent state and for poisson rate
	# 	ode_func_net = utils.create_net(dim * 2, args.latents * 2, 
	# 		n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

	# 	gen_ode_func = ODEFunc_w_Poisson(
	# 		input_dim = input_dim, 
	# 		latent_dim = args.latents * 2,
	# 		ode_func_net = ode_func_net,
	# 		lambda_net = lambda_net,
	# 		device = device).to(device)
	# else:

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
	obsrv_std = 0.01

	ode_func_net = utils.create_net(dim, params['var_dim'], 
		n_layers = params['n_layers'], n_units = params['rnn_dim'], nonlinear = nn.Tanh)

	gen_ode_func = ODEFunc(
		input_dim = input_dim, 
		latent_dim = params['var_dim'], 
		ode_func_net = ode_func_net,
		device = device).to(device)

	z0_diffeq_solver = None
	n_rec_dims = 40
	enc_input_dim = int(input_dim) * 2 # we concatenate the mask
	gen_data_dim = input_dim

	z0_dim = params['var_dim']
	# if args.poisson:
	# 	z0_dim += args.latents # predict the initial poisson rate

	# if args.z0_encoder == "odernn":
	ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
		n_layers = params['n_layers'], n_units = params['rnn_dim'], nonlinear = nn.Tanh)

	rec_ode_func = ODEFunc(
		input_dim = enc_input_dim, 
		latent_dim = n_rec_dims,
		ode_func_net = ode_func_net,
		device = device).to(device)

	z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", params['var_dim'], 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
	
	encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
		z0_dim = z0_dim, n_gru_units = params['rnn_dim'], device = device).to(device)

	# elif args.z0_encoder == "rnn":
	# 	encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
	# 		lstm_output_size = n_rec_dims, device = device).to(device)
	# else:
	# 	raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

	decoder = Decoder(params['var_dim'], gen_data_dim).to(device)

	diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', params['var_dim'], 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	model = LatentODE(
		input_dim = gen_data_dim, 
		latent_dim = params['var_dim'], 
		encoder_z0 = encoder_z0, 
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		use_poisson_proc = False, 
		use_binary_classif = False,
		linear_classifier = False,
		classif_per_tp = False,
		n_labels = 1,
		train_classif_w_reconstr = False
		).to(device)

	model.params = params
	model.params_str = params_str
	model.device = params['device']
	model.missing_mode = "block" if model.params["dynamic_missing"] else "block_all_feat"
	if model.params["dynamic_missing"]:
		if model.params["m_pattern"] == "camera_simulate":
			model.missing_mode = "camera_simulate"
		else:
			model.missing_mode = "player_wise"
	else:
		model.missing_mode = "all_player"
	model.dataset = params['dataset']
	model.n_features = params['n_features']

	return model
