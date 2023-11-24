CUDA_VISIBLE_DEVICES=0,1 python train.py --setting pipeline --epochs 50 --lr 0.0002 --batch_size 2048 
CUDA_VISIBLE_DEVICES=2,3 python train.py --setting pipeline --epochs 50 --lr 0.0002 --batch_size 2048 
