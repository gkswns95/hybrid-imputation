CUDA_VISIBLE_DEVICES=6 \
python train.py \
--trial 200 \
--dataset soccer \
--model dbhp \
--missing_pattern playerwise \
--missing_rate 0.5 \
--normalize \
--flip_pitch \
--team_size 11 \
--n_features 6 \
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
--cartesian_accel \
--fpe \
--fpi \
--bidirectional \
--pe_z_dim 16 \
--pi_z_dim 16 \
--rnn_dim 256 \
--hybrid_rnn_dim 128 \
--n_layers 2 \
--n_heads 4 \
--dropout 0.0 \
--deriv_accum \
--dynamic_hybrid \