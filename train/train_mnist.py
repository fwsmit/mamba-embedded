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
import onnx
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba


class Net(nn.Module):
    def __init__(self, d_model: int = 8, d_state: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(28, d_model, bias=False)   # row → d_model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=3,
            expand=2,
        )

        # self.mamba = Mamba(
        #     d_model=d_model, d_state=d_state,
        #     d_conv=3, dt_rank=1, expand=2,
        # )
        self.classifier = nn.Linear(d_model, 10, bias=False)  # d_model → 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28)   — standard torchvision MNIST format
        Returns:
            logits: (B, 10)
        """
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.mamba(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_onnx(onnx_path, comp_model, test_loader, device):
    # Load the ONNX model
    tmp_model = onnx.load(onnx_path)
    # Check that the model is well formed
    onnx.checker.check_model(tmp_model)
    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))

    comp_model.eval()
    ort_sess = ort.InferenceSession(onnx_path)
    i = 0
    valid = True
    atol = 1e-4
    with torch.no_grad():
        for data, _ in test_loader:
            # data, target = data.to(device), target.to(device)
            data_on_device = data.to(device)
            output_target = comp_model(data_on_device)

            # output = ort_sess.run(None, {'input.1': data.numpy()})
            output = ort_sess.run(None, {'input': data.numpy()})
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
    dataset1 = datasets.MNIST('../Datasets/', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../Datasets/', train=False,
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

        def _selective_scan_vectorized(u, delta, A, B, C, D=None,
                                       z=None, delta_bias=None,
                                       delta_softplus=False, return_last_state=False):
            """
            Vectorized pure-PyTorch drop-in for selective_scan_fn.
 
            The recurrence h_t = dA_t*h_{t-1} + dB_t*u_t expands to:
 
                h_t = sum_{s=0..t} (dA_{s+1}*...*dA_t) * dB_s * u_s
                    = P_t * cumsum(bu / P)[t]
 
            where P_s = dA_0 * dA_1 * ... * dA_s  (INCLUSIVE cumprod up to s).
 
            Verification:
              t=0: P_0*(bu_0/P_0) = bu_0                          = dB_0*u_0         ok
              t=1: P_1*(bu_0/P_0 + bu_1/P_1) = dA_1*bu_0 + bu_1                     ok
              t=2: P_2*(bu_0/P_0+bu_1/P_1+bu_2/P_2) = dA_1*dA_2*bu_0+dA_2*bu_1+bu_2 ok
 
            aten::cumprod is unsupported in ONNX, so we use the identity:
                cumprod(x) = exp(cumsum(log(x)))
            which is valid here because dA = exp(delta*A) > 0 always.
            """
            if delta_bias is not None:
                delta = delta + delta_bias.unsqueeze(-1)
            if delta_softplus:
                delta = F.softplus(delta)
 
            # Discretise: dA (B,d,L,N),  dB (B,d,L,N)
            dA = torch.exp(
                delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
            )
            dB = delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1)
 
            # Inclusive prefix products via exp(cumsum(log(dA))) — avoids cumprod
            # dA > 0 always, so log is safe
            P = torch.exp(torch.cumsum(torch.log(dA), dim=2))  # (B,d,L,N)
 
            # Vectorised scan: h = P * cumsum(bu / P, dim=L)
            bu = dB * u.unsqueeze(-1)                           # (B,d,L,N)
            h  = P * torch.cumsum(bu / P, dim=2)               # (B,d,L,N)
 
            # Output: y_t = sum_N h_t * C_t
            y = (h * C.permute(0, 2, 1).unsqueeze(1)).sum(-1)  # (B,d,L)
 
            if D is not None:
                y = y + D.unsqueeze(0).unsqueeze(-1) * u
            if z is not None:
                y = y * F.silu(z)
 
            if return_last_state:
                return y, h[:, :, -1, :]
            return y

        # use_fast_path=False alone is not enough — slow_forward still calls
        # causal_conv1d_fn and selective_scan_fn (custom CUDA extensions) via
        # module-level references that ONNX tracing cannot follow.
        # Temporarily null them out to force the pure-PyTorch fallback branches.
        import mamba_ssm.modules.mamba_simple as _mamba_mod
        _orig_ccf = _mamba_mod.causal_conv1d_fn
        _orig_ssf = _mamba_mod.selective_scan_fn

        _mamba_mod.causal_conv1d_fn = None
        _mamba_mod.selective_scan_fn = _selective_scan_vectorized
        model.mamba.use_fast_path = False

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
            opset_version=None,
            external_data=False,
            optimize=True,
            # dynamo=True,
        )

        # Restore for any subsequent inference / training
        _mamba_mod.causal_conv1d_fn = _orig_ccf
        # _mamba_mod.selective_scan_fn = _orig_ssf
        model.mamba.use_fast_path = True

        print("Testing onnx model")
        test_onnx(onnx_path, model, validate_loader, device)


if __name__ == '__main__':
    main()
