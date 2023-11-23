import torch 
import os
from vit import ViT
from time import time
from tqdm import tqdm
import numpy as np
import argparse
import wandb
import functools
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, transforms
import torch.nn.functional as F
from tiny_imagenet import TinyImageNetDataset

# Distributed imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# set cuda visible devices 0,2
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

# breakpoint()
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12441'

    # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Training loop

def validate(val_loader, model,rank, device):
    # Test the model
    criterion = nn.CrossEntropyLoss()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels).item()
            ddp_loss[0]+=val_loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            ddp_loss[1]+=correct
            ddp_loss[2]+=len(labels)
        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    return val_acc, val_loss,ddp_loss

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
    train_dataset = TinyImageNetDataset(root_dir='data/tiny-imagenet-200', transform=train_transforms)
    val_dataset = TinyImageNetDataset(root_dir='data/tiny-imagenet-200', mode='val', transform=val_transforms)
    
    return train_dataset, val_dataset

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    criterion = nn.CrossEntropyLoss()
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion (output, target)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
        wandb.log({"epoch": epoch, "loss": ddp_loss[0] / ddp_loss[1]})
def test(model, rank, world_size, test_loader,epoch):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
        wandb.log({"epoch": epoch,"test_loss": test_loss, "test_acc": 100. * ddp_loss[1] / ddp_loss[2]})
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if rank == 0:
        wandb.init(project="VIT_FSDP", name="2_gpu", config=args)
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                     transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                     transform=transform)

    dataset1, dataset2 = get_datasets()
    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    
    # train_loader = DataLoader(train_ds, batch_size=int(args.batch_size / args.world_size), shuffle=False, 
    #                           num_workers=4, sampler=DistributedSampler(train_ds))
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

    

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ViT(
        image_size = 224,
        patch_size = args.patch_size,
        num_classes = 200,
        dim = args.emb_dim,
        depth = 12,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(rank)

    model = FSDP(model)
    state = powerSGD.PowerSGDState(
        process_group=None, 
        matrix_approximation_rank=1,
        start_powerSGD_iter=10,
    )
    model.register_comm_hook(state, powerSGD.powerSGD_hook)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model ,rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader, epoch)
        optimizer.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()



if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser("ViT training script")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="test batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--emb_dim", type=int, default=768, help="embedding dimension")
    parser.add_argument("--setting", type=str, default="fsdp", help="Training setting")
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--save_model", action="store_true", help="Save model weights")
    
    args = parser.parse_args()
    
    wandb.init(project="VIT_FSDP", name="2_gpu", config=args)

    # if args.setting == "ddp":
    #     mp.spawn(ddp_train, args=(args.world_size, args), nprocs=args.world_size, join=True)
    if args.setting == "fsdp":
        mp.spawn(fsdp_main, args=(args.world_size, args), nprocs=args.world_size, join=True)