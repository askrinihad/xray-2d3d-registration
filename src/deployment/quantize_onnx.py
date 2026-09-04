"""
Apply dynamic quantization to the exported ONNX model -- reduces weight
precision (float32 -> int8) to shrink model size and speed up inference,
at some potential cost to accuracy. One of the standard steps for
preparing a model for real-time embedded deployment.

Note: dynamic quantization mainly shrinks weight storage; speed gains on
a small conv-heavy model like this can be modest compared to a
linear/FC-heavy model, since activations aren't pre-quantized. Still a
valid demonstration of the technique -- see profile_models.py for actual
measured numbers rather than assuming the speedup.

Requires results/pose_cnn.onnx -- run export_onnx.py first.

Run:
    python src/deployment/quantize_onnx.py
"""
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

FP32_PATH = "results/pose_cnn.onnx"
INT8_PATH = "results/pose_cnn_quantized.onnx"


def main():
    quantize_dynamic(
        model_input=FP32_PATH,
        model_output=INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    fp32_size = os.path.getsize(FP32_PATH) / 1024
    int8_size = os.path.getsize(INT8_PATH) / 1024
    print(f"FP32 model: {fp32_size:.1f} KB")
    print(f"INT8 model: {int8_size:.1f} KB")
    print(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
