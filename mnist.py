"""Train CIFAR10 with PyTorch."""
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR


if "darwin" in sys.platform.lower():
    # To avoid multiple runs of the model code
    # https://pythonspeed.com/articles/python-multiprocessing/
    import multiprocessing
    multiprocessing.set_start_method("fork")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--bs", default=128, type=int, help="batch size")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--epochs", default=60, type=int, help="number of epochs")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)


trainset = torchvision.datasets.CIFAR10(
    root=".", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, shuffle=True, num_workers=1, drop_last=True
)

validset = torchvision.datasets.CIFAR10(
    root=".", train=False, download=False, transform=transform_test
)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=100, shuffle=False, num_workers=1
)

# Model
print("==> Building model..")

net = Net()
net = net.to(device)
if device == "cuda":
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    [{"params": [param]} for param in net.parameters()],
    lr=args.lr,
    momentum=0.9,
    weight_decay=5e-4,
)
lr_scheduler = MultiStepLR(optimizer, [30, 45], 0.1)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    loss_sum, total, correct = 0, 0, 0
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    loss_avg = loss_sum / total
    accuracy = correct / total
    print(f"train: loss_avg = {loss_avg}, accuracy = {accuracy}")


def valid(epoch):
    net.eval()
    loss_sum, total, correct = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss_sum += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    loss_avg = loss_sum / total
    accuracy = correct / total
    print(f"valid: loss_avg = {loss_avg}, accuracy = {accuracy}")


for epoch in range(1, args.epochs):
    train(epoch)
    valid(epoch)
    lr_scheduler.step()

