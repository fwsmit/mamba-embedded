# Originally copied and modified from:
# https://github.com/pytorch/examples/blob/main/mnist/main.py
# under the following license:  BSD-3-Clause license

from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from .data import load_har_data, load_mnist_data, load_speechcommands_data
from .models import TinyMamba,TinyMamba2Multi, TinyMamba3Multi
from .onnx import export_onnx, test_onnx

dataset_dir = "./data"


def train(model, device, train_loader, optimizer, epoch, print_stats=False, log_interval=10, dry_run=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if print_stats and batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break


def test(model, device, test_loader, print_stats=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if (print_stats):
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--validate-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for validating (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        metavar="LR",
        help="learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    # parser.add_argument(
    #     "--save-model",
    #     action="store_true",
    #     default=False,
    #     help="For Saving the current Model",
    # )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="For Saving the current Model in ONNX format",
    )
    parser.add_argument(
        "--full-test-onnx",
        action="store_true",
        default=False,
        help="Test the onnx exported model fully against the pytorch model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load dataset based on DATASET environment variable
    dataset_type = os.environ.get("DATASET")
    print(f"Training with dataset: {dataset_type}")

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    validate_kwargs = {"batch_size": args.batch_size}
    validate_single_kwargs = {"batch_size": 1}
    test_kwargs = {"batch_size": args.validate_batch_size}
    if use_cuda:
        if dataset_type == "kws":
            num_workers = 8
        else:
            num_workers = 1
        cuda_kwargs = {"num_workers": num_workers, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validate_kwargs.update(cuda_kwargs)
        validate_single_kwargs.update(cuda_kwargs)


    if dataset_type == "mnist":
        output_size = 10
        input_dim = 28
        d_model = 8
        train_ds, val_ds, test_ds = load_mnist_data(dataset_dir)
    elif dataset_type == "har":
        output_size = 6
        input_dim = 57
        d_model = 16
        d_state = 8
        d_conv = 4
        expand = 2
        train_ds, val_ds, test_ds = load_har_data(dataset_dir)
    elif dataset_type == "kws":
        output_size = 35
        input_dim = 40
        d_model = 4
        train_ds, val_ds, test_ds = load_speechcommands_data(dataset_dir)
    else:
        sys.exit(f"Unknown dataset: {dataset_type}. Choose 'mnist', 'kws' or 'har'")

    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    _ = torch.utils.data.DataLoader(test_ds, **test_kwargs)  # Test loader is not used yet
    validate_loader = torch.utils.data.DataLoader(val_ds, **validate_kwargs)
    validate_loader_single = torch.utils.data.DataLoader(val_ds, **validate_single_kwargs)

    model_type = os.environ.get("MODEL")
    print(f"Training model: {model_type}, hidden dim: {d_model}")

    match model_type:
        case "mamba-1":
            model = TinyMamba(input_dim=input_dim,d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, output_size=output_size).to(device)
        case "mamba3":
            model = TinyMamba3Multi(input_dim=input_dim,d_model=d_model, d_state=d_state, output_size=output_size).to(device)
        case "mamba-2":
            model = TinyMamba2Multi(input_dim=input_dim,d_model=d_model, d_state=d_state, output_size=output_size).to(device)
        case _:
            sys.exit(
                "Please specify a correct model with the environment variable MODEL"
            )

    model_name = f"{dataset_type}-{model_type}"
    dry_run_name = ""
    model_dir = os.path.join("./src", "models")
    if args.dry_run:
        dry_run_name = "-dry"
        model_dir = "/tmp"

    onnx_path = os.path.join(model_dir, model_name + dry_run_name + ".onnx")
    pt_path = os.path.join(model_dir, model_name + dry_run_name + ".pt")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            True,
            log_interval=args.log_interval,
            dry_run=args.dry_run,
        )
        test(model, device, validate_loader, print_stats=True)
        scheduler.step()
        if args.dry_run:
            break

    torch.save(model, pt_path)

    if args.export_onnx:
        print("Exporting onnx model")
        export_onnx(model, dataset_type, onnx_path, device)
        print("Testing onnx model")
        test_onnx(onnx_path, model, validate_loader_single, device, args.full_test_onnx)
        print(
            f"ONNX model size: {os.path.getsize(onnx_path):,} bytes "
            f"({os.path.getsize(onnx_path) / 1024:.2f} KB)"
        )
        print("Exported everything to", model_dir)


if __name__ == "__main__":
    main()
