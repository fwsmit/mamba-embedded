#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "torch==2.10.0",
#   "onnxscript",
#   "torchvision",
#   "onnx==1.19.0",
# ]
# ///

# Originally copied and modified from: https://github.com/pytorch/examples/blob/main/mnist/main.py
# under the following license:  BSD-3-Clause license

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import onnxruntime as ort
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Flattening the 28x28 input image to 1D
        self.fc2 = nn.Linear(128, 64)        # Intermediate layer
        self.fc3 = nn.Linear(64, 10)         # Output layer for 10 classes (digits 0-9)
        self.dropout = nn.Dropout(0.2)       # Dropout for regularization

    def forward(self, x):
        x = torch.flatten(x, 1)              # Flatten the input without batch size
        x = F.relu(self.fc1(x))              # Activation after fc1
        x = self.dropout(x)                   # Apply dropout
        x = F.relu(self.fc2(x))              # Activation after fc2
        x = self.fc3(x)                       # Output logits
        output = F.log_softmax(x, dim=1)     # Apply softmax
        return output

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, 3)
#         self.conv2 = nn.Conv2d(8, 16, 3)
#         self.conv3 = nn.Conv2d(16, 24, 3)
#         self.norm1 = nn.BatchNorm2d(24)
#         self.dropout1 = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(24 * 22 * 22, 32)
#         self.fc2 = nn.Linear(32, 10)
#         self.norm2 = nn.BatchNorm1d(10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.norm1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.norm2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_onnx(onnx_path, comp_model, test_loader, device):
    comp_model.eval()
    ort_sess = ort.InferenceSession(onnx_path)
    i = 0
    valid = True
    atol = 1e-5
    with torch.no_grad():
        for data, _ in test_loader:
            # data, target = data.to(device), target.to(device)
            data_on_device = data.to(device)
            output_target = comp_model(data_on_device)

            # output = ort_sess.run(None, {'input.1': data.numpy()})
            output = ort_sess.run(None, {'onnx::Flatten_0': data.numpy()})
            target_np = output_target.cpu().numpy()
            are_similar = np.allclose(target_np, output, atol=atol)

            if not are_similar:
                print("Arrays are not similar for datapoint", i)
                print("target", output_target)
                print("result", output)
                valid = False
            i += 1

    if not valid:
        print("ONNX validation failed")
        exit(1)
    else:
        print("ONNX validation succeeded (e <", atol, ")")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--export-onnx', action='store_true', default=True,
                        help='For Saving the current Model in ONNX format')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    onnx_path = "./src/model/mnist.onnx"
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    validate_kwargs = {'batch_size': 1}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validate_kwargs.update(cuda_kwargs)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('/tmp/mnist-data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('/tmp/mnist-data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    validate_loader = torch.utils.data.DataLoader(dataset2, **validate_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model, "mnist.pt")

    if args.export_onnx:
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        torch.onnx.export(model, dummy_input, onnx_path,
                          verbose=False, opset_version=16, external_data=False)

        print("Testing onnx model")
        test_onnx(onnx_path, model, validate_loader, device)


if __name__ == '__main__':
    main()
