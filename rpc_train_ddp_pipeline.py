import torch 
import os
from vit import ViT
from pipe_vit import PipeViT
from time import time
from tqdm import tqdm
import numpy as np
import argparse
import wandb
from pdb import set_trace as stx

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, transforms

from tiny_imagenet import TinyImageNetDataset

# Distributed imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

## Pipeline import
from torch.distributed import rpc
import tempfile
from torch.distributed.pipeline.sync import Pipe

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12352'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Training loop
def train(train_loader, model, epoch_number, learning_rate, device):
    # Train the model
    num_epochs = epoch_number
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        start_time = time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            running_loss += loss.item()

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time

        # Validate every 3 epochs
        if (epoch + 1) % 3 == 0:
            torch.cuda.empty_cache()
            val_acc, val_loss = validate(val_loader, model, device)
            wandb.log({"epoch": epoch, "loss": running_loss / len(train_loader), "train_acc": train_accuracy, "val_acc": val_acc, "val_loss": val_loss})
            print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        else:
            wandb.log({"epoch": epoch, "epoch_time": epoch_time, "loss": running_loss / len(train_loader), "train_acc": train_accuracy})
            print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}")

def validate(val_loader, model, device):
    # Test the model
    criterion = nn.CrossEntropyLoss()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
    return val_acc, val_loss

def get_datasets():
    # Get dataset
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = TinyImageNetDataset(root_dir='./data', transform=train_transforms)
    val_dataset = TinyImageNetDataset(root_dir='./data', mode='val', transform=val_transforms)
    
    return train_dataset, val_dataset


def ddp_train(rank, world_size, args):
    setup(rank, world_size)

    # WandB stuff
    variant = 'B' if args.emb_dim == 768 else 'L'
    exp_name = f"ViT_{args.setting}_ep{args.epochs}_LR{args.lr}_BS{args.batch_size}_{args.patch_size}P_{variant}/emb{args.emb_dim}"
    wandb_config = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        )
    
    if rank == 0:
        wandb.init(project="ML710", entity="jameelhassan", name=exp_name, config=wandb_config)
    
    # create model and move it to GPU with id rank
    model = ViT(
        image_size = 224,
        patch_size = args.patch_size,
        num_classes = 200,
        dim = args.emb_dim,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    train_ds, val_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size / args.world_size), shuffle=False, 
                              num_workers=4, sampler=DistributedSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    num_epochs = args.epochs

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)
    for epoch in range(num_epochs):
        # Display progress bar only for rank 0
        if rank == 0:
            progress_bar = tqdm(total=len(train_loader), desc=f"Rank {rank}", position=0)
        else:
            progress_bar = None

        start_time = time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            running_loss += loss.item()
            if rank == 0:
                progress_bar.update(1)

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time

        if rank == 0:
            if (epoch + 1) % 1 == 0:
                torch.cuda.empty_cache()
                val_acc, val_loss = validate(val_loader, model, f'cuda:{str(rank)}')
                wandb.log({"epoch": epoch, "loss": running_loss / len(train_loader), "train_acc": train_accuracy, "val_acc": val_acc, "val_loss": val_loss})
                print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            else:
                wandb.log({"epoch": epoch, "epoch_time": epoch_time, "loss": running_loss / len(train_loader), "train_acc": train_accuracy})
                print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}\n")

    cleanup()

def model_ddp_pipe_rank(args, rank):
    model_layers = PipeViT(
            image_size = 224,
            patch_size = args.patch_size,
            num_classes = 200,
            dim = args.emb_dim,
            depth = args.depth,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    model_layers = [nn.Sequential(*layer).to(f"cuda:{str(idx + 2 * rank)}") for (idx, layer) in enumerate(model_layers)]
    return model_layers

def setup_ddp_pipe(rank, world_size):
    # Initialize process group and wrap model in DDP.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Need to use 'checkpoint=never' since as of PyTorch 1.8, Pipe checkpointing
# doesn't work with DDP.
from torch.distributed.pipeline.sync import Pipe

# Pipeline Training
def train_ddp_pipe(rank, world_size, args):
    def print_with_rank(msg):
        print('[RANK {}]: {}'.format(rank, msg))

    device = torch.device(2 * rank)
    variant = 'B' if args.emb_dim == 768 else 'L'
    exp_name = f"ViT_{args.setting}_ep{args.epochs}_LR{args.lr}_BS{args.batch_size}_{args.patch_size}P_{variant}/emb{args.emb_dim}"
    wandb_config = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        )
    
    # Get dataset
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = TinyImageNetDataset(root_dir='./data', transform=train_transforms)
    val_dataset = TinyImageNetDataset(root_dir='./data', mode='val', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

    ## DDP Pipeline 
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name)
        )
    )
    num_gpus = args.world_size
    model_layers = model_ddp_pipe_rank(args, rank)
    chunks = 8
    model = Pipe(torch.nn.Sequential(*model_layers), chunks=chunks, checkpoint="never")

    setup_ddp_pipe(rank, world_size)
    model = DDP(model)

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print_with_rank('Total parameters in model: {:,}'.format(get_total_params(model)))

    # Train the model
    num_epochs = args.epochs
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(num_epochs):
        start_time = time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(f"cuda:{0 + 2 * rank}"), labels.to(f"cuda:{1 + 2 * rank}")   # Pipeline is split across 2 GPUS strictly
            optimizer.zero_grad()

            outputs = model(images).local_value()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            running_loss += loss.item()

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time

        # # Validate every 3 epochs
        # if (epoch + 1) % 3 == 0:
        #     torch.cuda.empty_cache()
        #     val_acc, val_loss = validate(val_loader, model, device)
        #     # wandb.log({"epoch": epoch, "loss": running_loss / len(train_loader), "train_acc": train_accuracy, "val_acc": val_acc, "val_loss": val_loss})
        #     print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        # else:
        # wandb.log({"epoch": epoch, "epoch_time": epoch_time, "loss": running_loss / len(train_loader), "train_acc": train_accuracy})
        print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}\n")


if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser("ViT training script")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--depth", type=int, default=4, help="depth of transformer")
    parser.add_argument("--emb_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--setting", type=str, default="ddp_pipeline", help="Training setting")
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()
    
    # Common preps: Data, wandb configs
    variant = 'B' if args.emb_dim == 768 else 'L'
    exp_name = f"ViT_{args.setting}_ep{args.epochs}_LR{args.lr}_BS{args.batch_size}_{args.patch_size}P_{variant}/emb{args.emb_dim}"
    wandb_config = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        )
    
    # Get dataset
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = TinyImageNetDataset(root_dir='./data', transform=train_transforms)
    val_dataset = TinyImageNetDataset(root_dir='./data', mode='val', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

    ## Baseline training
    if args.setting == "baseline":
        model = ViT(
            image_size = 224,
            patch_size = args.patch_size,
            num_classes = 200,
            dim = args.emb_dim,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # # Sanity check
        # print("Running sanity validation check")
        # acc = validate(val_loader, model, device)

        #Train the model
        with wandb.init(project="ML710", entity="jameelhassan", name=exp_name, config=wandb_config):
            train(train_loader, model, epoch_number=args.epochs, learning_rate=args.lr, device=device)

    ## DDP training 
    elif args.setting == "ddp":
        mp.spawn(ddp_train, args=(args.world_size, args), nprocs=args.world_size, join=True)

    elif args.setting == "ddp_pipeline":
        mp.spawn(train_ddp_pipe, args=(args.world_size, args), nprocs=args.world_size, join=True)
