CUDA_VISIBLE_DEVICES=0,1 python train.py --setting ddp --world_size 2 --epochs 50 --lr 0.0002 --batch_size 320 
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 50 --lr 0.0002 --batch_size 160 
CUDA_VISIBLE_DEVICES=0,1 python train.py --setting pipeline --epochs 50 --lr 0.0002 --batch_size 2048 