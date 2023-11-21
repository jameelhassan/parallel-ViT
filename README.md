# parallel-ViT
Parallelizing a ViT model

### Setup

```
conda create -n parallel-vit python=3.8 -y
conda activate parallel-vit
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install wandb
pip install tqdm einops
```