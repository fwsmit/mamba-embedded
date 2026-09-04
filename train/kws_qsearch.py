"""Quantization search harness for KWS models using kws_dataset/*.pkl."""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.api.espdl_interface import (
    get_target_platform, generate_test_value, get_random_inputs,
)
from esp_ppq.executor import TorchExecutor
import esp_ppq.lib as PFL

from .quantize import (
    TARGET, collate_fn, QuantizationDivergedError,
    _graph_has_invalid_scales, get_input_quantization,
    quantize_dataset_to_bin, infer_input_shape,
)

REPO = Path(__file__).resolve().parent.parent
CALIB = REPO / "kws_dataset" / "calib.pkl"
VAL = REPO / "kws_dataset" / "val.pkl"


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_calib_loader(n_samples: int = 256) -> DataLoader:
    d = _load_pkl(CALIB)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    if n_samples < len(X):
        labels = d["y"]
        classes = sorted(set(labels.tolist()))
        groups = defaultdict(list)
        for i, lbl in enumerate(labels.tolist()):
            groups[lbl].append(i)
        chosen = []
        rng = np.random.default_rng(42)
        base, rem = divmod(n_samples, len(classes))
        bonus = set(rng.permutation(classes)[:rem].tolist())
        for cls in classes:
            arr = groups[cls]
            quota = base + (1 if cls in bonus else 0)
            if quota <= 0:
                continue
            if len(arr) <= quota:
                chosen.extend(arr)
            else:
                step = len(arr) / quota
                chosen.extend(arr[int(step * i)] for i in range(quota))
        indices = np.array(chosen, dtype=np.int64)
    else:
        indices = np.arange(len(X))
    calib_ds = TensorDataset(X[indices], y[indices])
    return DataLoader(calib_ds, batch_size=1, shuffle=False, drop_last=False)


def load_val():
    d = _load_pkl(VAL)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    return TensorDataset(X, y), np.asarray(d["y"])


def run_onnx_test(onnx_path, val_ds) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)
    preds, ys = [], []
    for data, target in loader:
        out = sess.run(None, {"input": data.numpy()})[0]
        preds.append(np.argmax(out, axis=1))
        ys.append(target.numpy())
    return np.mean(np.concatenate(preds) == np.concatenate(ys)) * 100.0


def run_torch_executor(quant_graph, val_ds, device, ratio=1.0) -> float:
    executor = TorchExecutor(graph=quant_graph, device=device)
    if ratio < 1.0:
        n = int(len(val_ds) * ratio)
        if n < 1:
            n = 1
        val_ds = torch.utils.data.Subset(val_ds, range(n))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)
    preds, ys = [], []
    for data, target in loader:
        out = executor.forward(inputs=data.to(device))[0].cpu().numpy()
        preds.append(np.argmax(out, axis=1))
        ys.append(target.numpy())
    return np.mean(np.concatenate(preds) == np.concatenate(ys)) * 100.0


def quantize(onnx_path, out_espdl, calib_loader, calib_steps, calib_algorithm,
             use_tqt, tqt_block, tqt_steps, tqt_lr, num_of_bits,
             dispatch_int16, device):
    import onnx
    m = onnx.load(str(onnx_path))
    input_shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]

    setting = QuantizationSettingFactory.espdl_setting()
    setting.quantize_activation_setting.calib_algorithm = calib_algorithm
    setting.quantize_parameter_setting.calib_algorithm = calib_algorithm
    if use_tqt:
        setting.tqt_optimization = True
        setting.tqt_optimization_setting.block_size = tqt_block
        setting.tqt_optimization_setting.steps = tqt_steps
        setting.tqt_optimization_setting.lr = tqt_lr
        setting.tqt_optimization_setting.collecting_device = device

    dispatching_override = None
    if dispatch_int16:
        plat16 = get_target_platform("esp32s3", 16)
        dispatching_override = {n: plat16 for n in dispatch_int16}

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(out_espdl),
        calib_dataloader=calib_loader,
        calib_steps=calib_steps,
        input_shape=input_shape,
        target="esp32s3",
        num_of_bits=num_of_bits,
        collate_fn=collate_loader_fn,        dispatching_override=dispatching_override,
        setting=setting,
        device=device,
        export_test_values=True,
        error_report=False,
        skip_export=False,
        verbose=0,
    )
    if _graph_has_invalid_scales(quant_graph):
        raise QuantizationDivergedError("NaN/Inf scales")
    return quant_graph


def collate_loader_fn(batch):
    x, _ = batch
    return x.to(DEVICE)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", default="x")
    ap.add_argument("--outroot", default=str(REPO / "out" / "kws_search"))
    ap.add_argument("--calib-alg", default="kl")
    ap.add_argument("--tqt", action="store_true")
    ap.add_argument("--tqt-block", type=int, default=512)
    ap.add_argument("--tqt-steps", type=int, default=3000)
    ap.add_argument("--tqt-lr", type=float, default=2e-4)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--calib-steps", type=int, default=256)
    ap.add_argument("--int16-list", default="[]")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-ratio", type=float, default=1.0)
    ap.add_argument("--dump-ops", action="store_true")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    calib_loader = load_calib_loader(args.calib_steps)
    val_ds, val_labels = load_val()
    onnx_path = Path(args.onnx)

    if args.dump_ops:
        # dump simplified op names of the quantized graph
        import onnx as _onnx
        m = _onnx.load(str(onnx_path))
        input_shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        setting = QuantizationSettingFactory.espdl_setting()
        setting.quantize_activation_setting.calib_algorithm = args.calib_alg
        setting.quantize_parameter_setting.calib_algorithm = args.calib_alg
        g = espdl_quantize_onnx(
            onnx_import_file=str(onnx_path),
            espdl_export_file=str(REPO / "out" / "kws_search" / f"{args.tag}.espdl"),
            calib_dataloader=calib_loader, calib_steps=args.calib_steps,
            input_shape=input_shape, target="esp32s3", num_of_bits=args.bits,
            collate_fn=collate_loader_fn, setting=setting, device=device,
            export_test_values=True, error_report=False, skip_export=False, verbose=0,
        )
        ops = {}
        for op in g.operations.values():
            ops[op.name] = op.type
        out = Path(args.outroot) / args.tag / "simplified_ops.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(ops, open(out, "w"), indent=1)
        print(f"dumped {len(ops)} ops -> {out}")
        return

    dispatch = json.loads(args.int16_list)
    out_dir = Path(args.outroot) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    espdl = out_dir / "model.espdl"

    try:
        qg = quantize(onnx_path, espdl, calib_loader, args.calib_steps,
                      args.calib_alg, args.tqt, args.tqt_block, args.tqt_steps,
                      args.tqt_lr, args.bits, dispatch, device)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    acc = run_torch_executor(qg, val_ds, device, args.eval_ratio)
    print(f"[quant] accuracy = {acc:.2f} %")

    configs = get_input_quantization(qg)
    quantize_dataset_to_bin(configs, val_ds, out_dir / "dataset.bin")

    json.dump({"accuracy": round(acc, 3), "tag": args.tag, "onnx": str(onnx_path),
               "int16": dispatch, "tqt": args.tqt, "calib": args.calib_alg,
               "calib_steps": args.calib_steps}, open(out_dir / "result.json", "w"), indent=2)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()