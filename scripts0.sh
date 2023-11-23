conda activate dist_vit2
python train_fsdp.py --epochs 50 --lr 0.0002 --batch_size 200 --setting "fsdp" --world_size 2
# CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --lr 0.00005 --batch_size 256 --emb_dim 1024