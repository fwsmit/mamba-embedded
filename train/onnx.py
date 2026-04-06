import onnxruntime as ort
import onnx
import torch
import numpy as np
import torch.nn.functional as F


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
    atol = 1e-4
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
