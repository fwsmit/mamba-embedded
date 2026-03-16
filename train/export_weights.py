#
# export_weights.py
# Convert PyTorch .pth checkpoints to C header weight arrays for MambaLite-Micro.
#
# Copyright (c) 2025 MambaLite-Micro Authors
# Licensed under the MIT License.

import torch
import numpy as np
from model import TinyMambaHAR
import onnx
import onnxruntime
from train import load_har_data, evaluate
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

def export_pth_to_onnx(pth_path, onnx_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyMambaHAR(input_dim=57, hidden_dim=64,
                         seq_len=10, num_classes=6).to(device)
    model.load_state_dict(torch.load(pth_path))
    model.eval()  # Set the model to evaluation mode

    # Define example input with the appropriate shape
    # Example: Batch size = 1, Sequence length = 10, Input dimension = 57
    dummy_input = torch.randn(1, 10, 57).to(device)

    # Export the model to ONNX format
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=18,
                      dynamo=True,
                      do_constant_folding=True,
                      # report=True,
                      )  # Use the appropriate opset version

def validate_model(pth_path, onnx_path, data_dir):
    import onnx

    # Load and inspect ONNX model
    model = onnx.load(onnx_path)

    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    
    X_train, y_train, X_test, y_test = load_har_data(data_dir)
    batch_size = 64
    test_ds = TensorDataset(X_test, y_test)
    print(X_test.shape)
    return


    test_loader = DataLoader(test_ds, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    hidden_dim = 64
    device = torch.device('cuda')
    
    model = TinyMambaHAR(input_dim=57, hidden_dim=hidden_dim,
                         seq_len=10, num_classes=6).to(device)
    model.load_state_dict(torch.load(pth_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"[Pth Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

# Load the ONNX model
    # onnx_path = "your_model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_path)

# Function to evaluate ONNX model
    def evaluate_onnx_model(ort_session, test_loader):
        all_predictions = []
        all_targets = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.numpy()  # Convert to numpy array
            targets = targets.numpy()

            # Perform inference
            ort_inputs = {ort_session.get_inputs()[0].name: inputs}
            ort_outs = ort_session.run(None, ort_inputs)
            
            # Assuming the output is logits for CrossEntropyLoss
            logits = ort_outs[0]  # Get the logits from the output
            loss = criterion(torch.tensor(logits), torch.tensor(targets))
            total_loss += loss.item()

            # Get predictions
            predictions = np.argmax(logits, axis=1)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        avg_loss = total_loss / len(test_loader)

        return avg_loss, accuracy

# Evaluate model
    test_loss, test_acc = evaluate_onnx_model(ort_session, test_loader)
    print(f"[ONNX Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    data_dir = r'../Datasets/har-uci-dataset/UCI HAR Dataset/'
    export_pth_to_onnx("../Models/MambaLite-Micro/linear_har_model.pth", "./src/model/linear_har_model.onnx")
    # validate_model("../Models/MambaLite-Micro/mamba_har_model.pth", "../Models/MambaLite-Micro/mamba_har_model.onnx", data_dir)
    # export_pth_to_header("path/to/model.pth", "path/to/output/mamba_weights.h")
