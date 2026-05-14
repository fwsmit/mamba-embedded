import onnxruntime as ort
import onnx
import torch
import numpy as np
import torch.nn.functional as F
from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref
import mamba_ssm.modules.mamba_simple as _mamba_mod
import mamba_ssm.ops.triton.layernorm_gated as _ln_mod
from .mamba_cpu_funcs import _selective_scan_vectorized
from .data import get_data_input_size


def test_onnx(onnx_path, comp_model, test_loader, device, full_test):
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
    atol = 2e-3
    with torch.no_grad():
        for data, _ in test_loader:
            # data, target = data.to(device), target.to(device)
            data_on_device = data.to(device)
            output_target = comp_model(data_on_device)

            # data_pad = F.pad(data, (0,0,0,11))
            # output = ort_sess.run(None, {'input.1': data.numpy()})
            output = ort_sess.run(None, {"input": data.numpy()})
            target_np = output_target.cpu().numpy()
            are_similar = np.allclose(target_np, output, atol=atol)

            if not are_similar:
                print("Arrays are not similar for datapoint", i)
                print("target", output_target)
                print("result", output)
                print("max difference", np.max(np.absolute(output_target.cpu().numpy()-np.array(output))))
                valid = False
            i += 1

            if not full_test and i > 100:
                break

    if not valid:
        print("ONNX validation failed")
        exit(1)
    else:
        print("ONNX validation succeeded (e <", atol, ")")


# Capture original functions to place back later
_orig_ccf = _mamba_mod.causal_conv1d_fn
_orig_rms = _ln_mod.rmsnorm_fn
_orig_softplus = F.softplus           # keep a reference for put_back


# Different implementation of softplus, so that softplus does not end up in the
# ONNX graph (unsupported by esp-dl)
def _softplus_stable(x, beta=1, threshold=20):
    """
    Numerically stable softplus using only ONNX-exportable ops supported by esp-dl.

    softplus(x) = relu(x) + ln(1 + e^{-|x|})

    |x| is computed as relu(x) + relu(-x) to avoid the Abs ONNX node,
    which esp-dl does not support.
    """
    if beta != 1:
        x = x * beta

    # |x| = relu(x) + relu(-x)  — avoids the Abs node entirely
    neg_abs_x = -(F.relu(x) + F.relu(-x))   # = -|x|
    result = F.relu(x) + torch.log1p(torch.exp(neg_abs_x))

    if beta != 1:
        result = result / beta

    return result


def replace_unexportable_functions():
    # Replace mamba GPU kernels
    _mamba_mod.causal_conv1d_fn = None
    _mamba_mod.selective_scan_fn = _selective_scan_vectorized
    _ln_mod.rmsnorm_fn = rms_norm_ref
    # Replace softplus with a stable, ONNX-exportable equivalent
    F.softplus = _softplus_stable


def put_back_unexportable_functions():
    # Put back mamba GPU kernels
    _mamba_mod.causal_conv1d_fn = _orig_ccf
    _ln_mod.rmsnorm_fn = _orig_rms
    # Restore original softplus
    F.softplus = _orig_softplus


def export_onnx(model, dataset_type, onnx_path, device):
    input_size = get_data_input_size(dataset_type)
    # dummy_input is already defined above based on dataset_type
    if dataset_type == "mnist":
        # For MNIST: (1, 784, 1) - flattened 28x28 image as sequence
        # dummy_input = torch.randn(1, 784, 1, device=device)
        dummy_input = torch.randn(1, 1, input_size, input_size, device=device)
    if dataset_type == "kws":
        dummy_input = torch.randn(1, 51, input_size, device=device)
    else:
        # For HAR: (1, 561, 1) - 561 features as sequence
        dummy_input = torch.randn(1, 10, input_size, device=device)

    # Patch out some triton kernels with CPU implementations
    replace_unexportable_functions()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        opset_version=18,
        external_data=False,
        optimize=True,
        dynamo=False,
        do_constant_folding=True,
    )
    put_back_unexportable_functions()

