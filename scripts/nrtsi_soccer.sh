#!/bin/bash
python train.py \
--trial 9999 \
--dataset soccer \
--model nrtsi \
--target_type imputation \
--missing_pattern uniform \
--n_players 11 \
--n_features 2 \
--train_metrica \
--valid_metrica \
--n_max_time_scale 100 \
--time_enc_dim 8 \
--att_dim 128 \
--model_dim 1024 \
--inner_dim 2048 \
--time_dim 72 \
--expand_dim 5 \
--n_layers 8 \
--n_heads 12 \
--n_max_level 4 \
--cartesian_accel \
--use_ta \
--use_mask \
--xy_sort \
--stochastic \
--n_epochs 3100 \
--start_lr 1e-4 \
--min_lr 1e-3 \
--batch_size 16 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--clip 10 \
--max_iter_num 80000 \
--cuda \
--normalize \
--flip_pitch \