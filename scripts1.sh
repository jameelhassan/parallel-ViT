CUDA_VISIBLE_DEVICES=0,2 python train_quntization.py --epochs 35 --lr 0.0002 --batch_size 320 --setting "ddp" --world_size 2
# CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --lr 0.00005 --batch_size 256 --emb_dim 1024