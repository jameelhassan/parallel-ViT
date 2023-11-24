FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN python -m pip install --upgrade pip
RUN python -m pip install wandb scipy tqdm einops imageio
RUN python -m pip install torchgpipe
