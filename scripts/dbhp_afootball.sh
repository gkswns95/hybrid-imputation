CUDA_VISIBLE_DEVICES=0 \
python train.py \
--trial 600 \
--dataset afootball \
--model dbhp \
--missing_pattern playerwise \
--missing_rate 0.5 \
--team_size 6 \
--n_features 6 \
--window_size 50 \
--window_stride 5 \
--n_epochs 100 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 32 \
--print_every_batch 50 \
--save_every_epoch 100 \
--seed 100 \
--cuda \
--bidirectional \
--fpe \
--fpi \
--pe_z_dim 16 \
--pi_z_dim 16 \
--rnn_dim 256 \
--hybrid_rnn_dim 128 \
--n_layers 2 \
--n_heads 4 \
--dropout 0.0 \
--cartesian_accel \
--deriv_accum \
--dynamic_hybrid \