"""
quantize.py — ONNX export and ORT dynamic int8 quantization utilities.

Equivalent to: quantize_(model, Int8WeightOnlyConfig(version=2), filter_fn=_target_linear_only)
  where _target_linear_only = isinstance(module, nn.Linear) and module.in_features > 1
"""

import torch
import torch.nn as nn
import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


def export_fp32_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_names: list[str] = ["input"],
    output_names: list[str] = ["output"],
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> None:
    """Export a trained fp32 model to ONNX."""
    model.eval()
    dynamic_axes = (
        {name: {0: "batch"} for name in input_names + output_names}
        if dynamic_batch else None
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    print(f"Exported fp32 ONNX → {output_path}")


def preprocess_onnx(input_path: str, output_path: str) -> None:
    """
    Run ORT shape inference and graph optimisation.
    Required before quantization so per-channel scales can be computed correctly.

    skip_symbolic_shape=True: ORT's symbolic shape inference crashes on Loop nodes
    whose optional inputs are empty strings (a known ORT bug). ONNX-level shape
    inference (skip_onnx_shape=False) still runs and is sufficient for per-channel
    weight quantization.
    """
    quant_pre_process(
        input_model_path=input_path,
        output_model_path=output_path,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
        verbose=0,
    )
    print(f"Pre-processed ONNX → {output_path}")


def get_nodes_to_exclude_by_in_features(model_path: str, min_in_features: int = 2) -> list[str]:
    """
    Return names of MatMul/Gemm nodes whose weight in_features < min_in_features.
    Mirrors torchao's filter_fn: isinstance(module, nn.Linear) and module.in_features > 1.

    Weight layout conventions:
      MatMul: weight is [..., in_features, out_features]
      Gemm:   weight is [out_features, in_features] when transB=1 (torch default)
    """
    m = onnx.load(model_path)
    init_shape: dict[str, list[int]] = {
        init.name: list(init.dims) for init in m.graph.initializer
    }

    exclude = []
    for node in m.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        if len(node.input) < 2:
            continue

        weight_name = node.input[1]
        shape = init_shape.get(weight_name)
        if shape is None:
            continue  # dynamic weight — skip

        if node.op_type == "MatMul":
            in_features = shape[-2] if len(shape) >= 2 else 1
        else:  # Gemm
            transB = next(
                (attr.i for attr in node.attribute if attr.name == "transB"), 0
            )
            in_features = shape[1] if transB else shape[0]

        if in_features < min_in_features:
            print(f"  Excluding node '{node.name}' — in_features={in_features}")
            exclude.append(node.name)

    return exclude


def quantize_onnx_int8(
    input_path: str,
    output_path: str,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    """
    Apply ORT dynamic int8 quantization, equivalent to Int8WeightOnlyConfig(version=2):
      - Weights: int8, per-channel symmetric (zero_point=0)
      - Activations: int8 per-tensor, quantized dynamically at runtime
      - Output ops: QLinearMatMul
    """
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=nodes_to_exclude or [],
        # optimize_model=False,  # already done in preprocess_onnx
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
    )
    # quantize_dynamic(
    #     model_input=input_path,
    #     model_output=output_path,
    #     op_types_to_quantize=["MatMul", "Gemm"],
    #     weight_type=QuantType.QInt8,
    #     per_channel=True,
    #     reduce_range=False,
    #     nodes_to_exclude=nodes_to_exclude or [],
    #     # optimize_model=False,  # already done in preprocess_onnx
    #     extra_options={
    #         "WeightSymmetric": True,
    #         "ActivationSymmetric": False,
    #     },
    # )
    print(f"Quantized int8 ONNX → {output_path}")


def summarise_quantization(model_path: str) -> None:
    """Print op counts and number of int8 weight tensors in an ONNX graph."""
    m = onnx.load(model_path)

    op_counts: dict[str, int] = {}
    for node in m.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"\nONNX op summary for '{model_path}':")
    for op, count in sorted(op_counts.items()):
        print(f"  {op:30s} × {count}")

    int8_weights = [
        init for init in m.graph.initializer
        if init.data_type == onnx.TensorProto.INT8
    ]
    print(f"  INT8 weight tensors: {len(int8_weights)}")


def verify_quantization(
    fp32_path: str,
    int8_path: str,
    dummy_input: np.ndarray,
    input_name: str = "input",
) -> None:
    """Run both models through ORT and print max/mean abs errors."""
    import onnxruntime as ort

    def _run(path, x):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess.run(None, {input_name: x})[0]

    out_fp32 = _run(fp32_path, dummy_input)
    out_int8 = _run(int8_path, dummy_input)

    print(f"\nQuantization error ({fp32_path} vs {int8_path}):")
    print(f"  Max  abs error: {np.abs(out_int8 - out_fp32).max():.6f}")
    print(f"  Mean abs error: {np.abs(out_int8 - out_fp32).mean():.6f}")
