for X in {0..8}
do
CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X --twolayer

done