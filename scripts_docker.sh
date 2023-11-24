# CUDA_VISIBLE_DEVICES=2,3 python train_fsdp.py --epochs 50 --lr 0.0002 --batch_size 200 --setting "fsdp" --world_size 2
# cd /k8s_data
wandb login fa0767adc156a87ed43a394680774f3116fc3ed2
# CUDA_VISIBLE_DEVICES=2,3 python train_quantization.py --epochs 50 --lr 0.0002 --batch_size 320 --setting "ddp" --world_size 2
CUDA_VISIBLE_DEVICES=2,3 python train_powersgd.py --epochs 50 --lr 0.0002 --batch_size 320 --setting "ddp" --world_size 2
CUDA_VISIBLE_DEVICES=2,3 python train_fsdp.py --epochs 50 --lr 0.0002 --batch_size 160 --setting "fsdp" --world_size 2

# python train_fsdp.py --epochs 50 --lr 0.0002 --batch_size 200 --setting "fsdp" --world_size 2
# CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --lr 0.00005 --batch_size 256 --emb_dim 1024