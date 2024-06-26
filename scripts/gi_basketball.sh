python train.py \
--trial 9998 \
--dataset basketball \
--model ours \
--target_type imputation \
--n_players 5 \
--n_features 6 \
--train_nba \
--valid_nba \
--pe_z_dim 64 \
--pi_z_dim 32 \
--rnn_dim 256 \
--hybrid_rnn_dim 512 \
--n_layers 2 \
--n_heads 4 \
--dropout 0.0 \
--physics_loss \
--cartesian_accel \
--transformer \
--fpe \
--fpi \
--dynamic_hybrid \
--bidirectional \
--dynamic_missing \
--stochastic \
--seperate_learning \
--n_epochs 100 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 16 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda \
--normalize \
--flip_pitch \