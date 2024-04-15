python train.py \
--trial 4002 \
--dataset soccer \
--model brits \
--target_type imputation \
--missing_pattern camera \
--n_players 11 \
--n_features 2 \
--train_metrica \
--valid_metrica \
--rnn_dim 512 \
--dropout 0.0 \
--cartesian_accel \
--xy_sort \
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