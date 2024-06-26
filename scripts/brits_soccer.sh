CUDA_VISIBLE_DEVICES=0 \
python train.py \
--trial 99 \
--dataset soccer \
--model brits \
--missing_pattern camera \
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
--batch_size 32 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda \
--rnn_dim 512 \
--dropout 0.0 \