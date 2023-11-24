import torch 
import os
from vit import ViT
from time import time
from tqdm import tqdm
import numpy as np
import argparse
import wandb

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
from pipe_vit import PipeViT

## GPipe import
from pipe_vit import GPipeViT
from torchgpipe import GPipe


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Training loop
def train(train_loader, model, epoch_number, learning_rate, device):
    # Train the model
    num_epochs = epoch_number
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    iters = 0
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
            wandb.log({"loss": loss, "iter": iters})
            iters += 1

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time

        print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}")
        
    return model

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
    train_dataset = TinyImageNetDataset(root_dir='/app/data', transform=train_transforms)
    val_dataset = TinyImageNetDataset(root_dir='/app/data', mode='val', transform=val_transforms)
    
    return train_dataset, val_dataset


def ddp_train(rank, world_size, args):
    setup(rank, world_size)

    # WandB stuff
    variant = 'B' if args.emb_dim == 768 else 'L'
    exp_name = f"ViT_{args.setting}_ep{args.epochs}_LR{args.lr}_BS{args.batch_size}_{args.patch_size}P_{variant}/emb{args.emb_dim}_pipe{args.pipe_chunks}"
    wandb_config = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        )
    
    if rank == 0:
        wandb.init(project="ML710-new", entity="jameelhassan", name=exp_name, config=wandb_config)
    
    # create model and move it to GPU with id rank
    model = ViT(
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

    train_ds, val_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size / args.world_size), shuffle=False, 
                              num_workers=args.num_workers, sampler=DistributedSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers)

    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    num_epochs = args.epochs

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)
    iters = 0
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

            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            running_loss += loss.item()
            if rank == 0:
                progress_bar.update(1)
                wandb.log({"loss": loss, "iter": iters})
                iters += 1

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time

        if rank == 0:
            print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}\n")

    cleanup()

def model_pipe_rank(args):
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

    model_layers = [nn.Sequential(*layer).to(f"cuda:{str(rank)}") for (rank, layer) in enumerate(model_layers)]
    return model_layers

# Pipeline Training
def train_pipe(train_loader, model, epoch_number, learning_rate):
    # Train the model
    num_epochs = epoch_number
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    iters = 0
    for epoch in range(num_epochs):
        start_time = time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to("cuda:0"), labels.to("cuda:1")
            optimizer.zero_grad()

            outputs = model(images).local_value()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            running_loss += loss.item()
            wandb.log({"loss": loss, "iter": iters})
            iters += 1

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time

        print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}")

def validate_pipe(val_loader, model):
    # Test the model for pipeline on 2 GPUs
    criterion = nn.CrossEntropyLoss()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to("cuda:0"), labels.to("cuda:1")
            outputs = model(images).local_value()
            val_loss = criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
    return val_acc, val_loss

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

    setup_ddp_pipe(rank, world_size)

    device = 2 * rank
    variant = 'B' if args.emb_dim == 768 else 'L'
    exp_name = f"ViT_{args.setting}_ep{args.epochs}_LR{args.lr}_BS{args.batch_size}_{args.patch_size}P_{variant}/emb{args.emb_dim}_pipe{args.pipe_chunks}"
    wandb_config = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        )
    
    if rank == 0:
        wandb.init(project="ML710-new", entity="jameelhassan", name=exp_name, config=wandb_config)
    
    num_gpus = args.world_size
    # Get dataset
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = TinyImageNetDataset(root_dir='/app/data', transform=train_transforms)
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size / num_gpus), shuffle=False, 
                              num_workers=args.num_workers, sampler=train_sampler)

    ## DDP Pipeline 
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

    model_layers = model_ddp_pipe_rank(args, rank)
    chunks = args.pipe_chunks
    model = Pipe(torch.nn.Sequential(*model_layers), chunks=chunks, checkpoint="never")

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
    iters = 0
    for epoch in range(num_epochs):
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
            images, labels = images.to(f"cuda:{0 + device}"), labels.to(f"cuda:{1 + device}")   # Pipeline is split across 2 GPUS strictly
            optimizer.zero_grad()

            outputs = model(images).local_value()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            running_loss += loss.item()
            if rank == 0:
                progress_bar.update(1)
                wandb.log({"loss": loss, "iter": iters})
                iters += 1

        train_accuracy = 100 * correct / total
        epoch_time = time() - start_time
        
        if rank == 0:
            print(f"Epoch [{epoch}/{num_epochs-1}] {epoch_time:.2f}secs, Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, iters: {epoch*len(train_loader)}\n")

def choose_pipe_split(split_id):
    '''
    Hardcoded splits for GPipe
    '''
    if split_id == 1:
        balance = [4, 3, 3, 4]
    elif split_id == 2:
        balance = [5, 3, 3, 3]
    elif split_id == 3:
        balance = [3, 3, 4, 4]

    return balance


if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser("ViT training script")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=160, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument("--depth", type=int, default=12, help="depth of transformer")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--emb_dim", type=int, default=768, help="embedding dimension")
    parser.add_argument("--setting", type=str, default="baseline", help="Training setting")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--pipe_split", type=int, default=1, help="Version of splits for pipeline")
    parser.add_argument("--pipe_chunks", type=int, default=8, help="Number of chunks for pipeline")
    args = parser.parse_args()
    
    os.environ['WANDB_API_KEY'] = '3ce0ed4f4d53f1fd9fd94fd418465c4a15d58583'
    ## Data and wandb inits
    variant = 'B' if args.emb_dim == 768 else 'L'
    exp_name = f"ViT_{args.setting}_ep{args.epochs}_LR{args.lr}_BS{args.batch_size}_{args.patch_size}P_{variant}/emb{args.emb_dim}_pipe{args.pipe_chunks}"
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
    train_dataset = TinyImageNetDataset(root_dir='/app/data', transform=train_transforms)
    val_dataset = TinyImageNetDataset(root_dir='/app/data', mode='val', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers)
    
    ## Baseline training
    if args.setting == "baseline":
        model = ViT(
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # # Sanity check
        # print("Running sanity validation check")
        # acc, loss = validate(val_loader, model, device)

        #Train the model
        with wandb.init(project="ML710-new", entity="jameelhassan", name=exp_name, config=wandb_config):
            model = train(train_loader, model, epoch_number=args.epochs, learning_rate=args.lr, device=device)

    ## DDP training 
    elif args.setting == "ddp":
        mp.spawn(ddp_train, args=(args.world_size, args), nprocs=args.world_size, join=True)
    
    ## Pipeline training
    elif args.setting == "pipeline":
        tmpfile = tempfile.NamedTemporaryFile()
        rpc.init_rpc(
            name="worker",
            rank=0,
            world_size=1,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method="file://{}".format(tmpfile.name)
            )
        )

        model_layers = model_pipe_rank(args)
        model = Pipe(torch.nn.Sequential(*model_layers), chunks=args.pipe_chunks)

        # # Sanity val check
        # print("Sanity validation check on pipeline")
        # acc = validate_pipe(val_loader, model)

        with wandb.init(project="ML710-new", entity="jameelhassan", name=exp_name, config=wandb_config):
            train_pipe(train_loader, model, epoch_number=args.epochs, learning_rate=args.lr)

    elif args.setting == "ddp_pipeline":
        mp.spawn(train_ddp_pipe, args=(args.world_size, args), nprocs=args.world_size, join=True)
    
    ## Pipeline training
    elif args.setting == "gpipe":
        model = GPipeViT(
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
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        balance = choose_pipe_split(args.pipe_split)
        wandb_config["pipe_split"] = balance
        wandb_config["pipe_chunks"] = args.pipe_chunks
        exp_name = exp_name + f"_split{args.pipe_split}_chunks{args.pipe_chunks}"

        model = GPipe(model, balance=balance, chunks=args.pipe_chunks)
        with wandb.init(project="ML710-new", entity="jameelhassan", name=exp_name, config=wandb_config):
            model = train(train_loader, model, epoch_number=args.epochs, learning_rate=args.lr, device=device)