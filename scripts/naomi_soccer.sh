CUDA_VISIBLE_DEVICES=0 \
python train.py \
--trial 600 \
--dataset soccer \
--model naomi \
--missing_pattern uniform \
--missing_rate 0.5 \
--normalize \
--flip_pitch \
--player_order xy_sort \
--team_size 11 \
--n_features 2 \
--window_size 200 \
--window_stride 5 \
--n_epochs 100 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 64 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda \
--rnn_dim 300 \
--dec1_dim 200 \
--dec2_dim 200 \
--dec4_dim 200 \
--dec8_dim 200 \
--dec16_dim 200 \
--n_layers 2 \
--n_highest 4 \
--stochastic \