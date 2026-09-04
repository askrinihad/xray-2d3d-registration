"""
Export the trained PoseCNN to ONNX format -- the standard interchange
format for deploying a PyTorch model to a different inference engine
(ONNX Runtime, TensorRT, etc.), independent of the training framework.

Requires results/pose_cnn.pt -- train it with train_pose_cnn.py (or the
Colab notebook) first, and make sure the checkpoint is in results/.

Run:
    python src/deployment/export_onnx.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "registration"))
from pose_cnn import PoseCNN

CHECKPOINT = "results/pose_cnn.pt"
ONNX_PATH = "results/pose_cnn.onnx"


def main():
    model = PoseCNN()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 200, 200)

    os.makedirs("results", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["image"],
        output_names=["pose"],
        dynamic_axes={"image": {0: "batch_size"}, "pose": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,  # force the older, more mature exporter -- the newer
                        # "dynamo" exporter (default in recent PyTorch) can
                        # produce graphs that trip up onnxruntime's
                        # quantization shape inference
    )
    print(f"Exported ONNX model to {ONNX_PATH}")

    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structural check passed.")


if __name__ == "__main__":
    main()