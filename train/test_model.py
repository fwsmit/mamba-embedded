import os
import torch
from .onnx import confusion_matrix_onnx
from .kws_dataset_gen import IDX2LABEL
from .data import SpeechCommandsMFCC
onnx_path = os.path.join("src", "models", "kws-mamba-1.onnx")
# onnx_path = "/tmp/kws-mamba-1-dry.onnx"

val_ds = SpeechCommandsMFCC.load("data/speech_commands_v0.02_augmented" + "/val.pkl")

validate_single_kwargs = {"batch_size": 1}
cuda_kwargs = {
    "num_workers": 1,
    "pin_memory": True,
}
validate_single_kwargs.update(cuda_kwargs)
validate_loader_single = torch.utils.data.DataLoader(val_ds, **validate_single_kwargs)

_ = confusion_matrix_onnx(
    onnx_path=onnx_path,
    test_loader=validate_loader_single,
    idx2label=IDX2LABEL,
)
