"""
Profile inference latency and model size across three versions of the
pose-prediction model: native PyTorch, exported ONNX (FP32), and
quantized ONNX (INT8). This is the "does deployment optimization
actually help" check -- respects the job's real interest in measured
latency/memory numbers, not assumed ones.

Requires results/pose_cnn.pt, results/pose_cnn.onnx, and
results/pose_cnn_quantized.onnx -- run export_onnx.py and
quantize_onnx.py first.

Run:
    python src/deployment/profile_models.py
"""
import os
import sys
import time

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "registration"))
from pose_cnn import PoseCNN

N_WARMUP = 10
N_RUNS = 100


def profile_pytorch():
    model = PoseCNN()
    model.load_state_dict(torch.load("results/pose_cnn.pt", map_location="cpu"))
    model.eval()
    x = torch.randn(1, 1, 200, 200)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        start = time.perf_counter()
        for _ in range(N_RUNS):
            model(x)
        elapsed = time.perf_counter() - start

    n_params = sum(p.numel() for p in model.parameters())
    return elapsed / N_RUNS * 1000, n_params  # ms per inference


def profile_onnx(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, 200, 200).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(N_WARMUP):
        session.run(None, {input_name: x})
    start = time.perf_counter()
    for _ in range(N_RUNS):
        session.run(None, {input_name: x})
    elapsed = time.perf_counter() - start

    return elapsed / N_RUNS * 1000  # ms per inference


def main():
    print(f"Profiling over {N_RUNS} runs (after {N_WARMUP} warmup runs), "
          f"CPU-only for a fair apples-to-apples comparison...\n")

    pt_ms, n_params = profile_pytorch()
    onnx_fp32_ms = profile_onnx("results/pose_cnn.onnx")
    onnx_int8_ms = profile_onnx("results/pose_cnn_quantized.onnx")

    pt_size = os.path.getsize("results/pose_cnn.pt") / 1024
    onnx_fp32_size = os.path.getsize("results/pose_cnn.onnx") / 1024
    onnx_int8_size = os.path.getsize("results/pose_cnn_quantized.onnx") / 1024

    print(f"Model parameters: {n_params:,}\n")
    print(f"{'variant':<20} {'latency (ms)':>14} {'size (KB)':>12}")
    print(f"{'PyTorch (fp32)':<20} {pt_ms:>14.3f} {pt_size:>12.1f}")
    print(f"{'ONNX (fp32)':<20} {onnx_fp32_ms:>14.3f} {onnx_fp32_size:>12.1f}")
    print(f"{'ONNX (int8)':<20} {onnx_int8_ms:>14.3f} {onnx_int8_size:>12.1f}")

    print(f"\nONNX vs PyTorch speedup: {pt_ms / onnx_fp32_ms:.2f}x")
    print(f"INT8 vs FP32 ONNX speedup: {onnx_fp32_ms / onnx_int8_ms:.2f}x")
    print(f"INT8 vs FP32 ONNX size reduction: {(1 - onnx_int8_size / onnx_fp32_size) * 100:.1f}%")


if __name__ == "__main__":
    main()
