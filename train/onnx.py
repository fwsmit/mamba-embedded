import onnxruntime as ort
import onnx
import torch
import numpy as np
import torch.nn.functional as F
from .selective_scan import _selective_scan_vectorized


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


def export_onnx(model, dataset_type, onnx_path, device):
    # dummy_input is already defined above based on dataset_type
    if dataset_type == "mnist":
        # For MNIST: (1, 784, 1) - flattened 28x28 image as sequence
        # dummy_input = torch.randn(1, 784, 1, device=device)
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
    if dataset_type == "kws":
        dummy_input = torch.randn(1, 51, 40, device=device)
    else:
        # For HAR: (1, 561, 1) - 561 features as sequence
        dummy_input = torch.randn(1, 10, 57, device=device)


    # use_fast_path=False alone is not enough — slow_forward still calls
    # causal_conv1d_fn and selective_scan_fn (custom CUDA extensions) via
    # module-level references that ONNX tracing cannot follow.
    # Temporarily null them out to force the pure-PyTorch fallback branches.
    import mamba_ssm.modules.mamba_simple as _mamba_mod

    _orig_ccf = _mamba_mod.causal_conv1d_fn
    # _orig_ssf = _mamba_mod.selective_scan_fn

    _mamba_mod.causal_conv1d_fn = None
    _mamba_mod.selective_scan_fn = _selective_scan_vectorized
    # model.mamba.use_fast_path = False

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        opset_version=17,
        external_data=False,
        optimize=True,
        dynamo=False,
    )

    # Restore for any subsequent inference / training
    _mamba_mod.causal_conv1d_fn = _orig_ccf
    # _mamba_mod.selective_scan_fn = _orig_ssf
    # model.mamba.use_fast_path = True

