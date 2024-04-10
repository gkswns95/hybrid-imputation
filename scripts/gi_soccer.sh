#!/bin/bash
python train.py \
--trial 7238 \
--dataset soccer \
--model graph_imputer \
--target_type imputation \
--n_players 11 \
--n_features 6 \
--train_metrica \
--valid_metrica \
--pe_z_dim 64 \
--pi_z_dim 32 \
--rnn_dim 64 \
--hybrid_rnn_dim 512 \
--n_layers 2 \
--n_heads 4 \
--dropout 0.0 \
--physics_loss \
--cartesian_accel \
--fpe \
--fpi \
--train_hybrid \
--bidirectional \
--dynamic_missing \
--avg_length_loss \
--n_epochs 1000 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 64 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--var_dim 16 \
--kld_weight_float 0.01 \
--cuda \
--normalize \
--flip_pitch \
--missing_prob_float 0.6 \
--m_pattern, camera_simulate