"""
Validate that ONNX export and INT8 quantization didn't meaningfully hurt
accuracy, by comparing pose predictions from the original PyTorch model,
the exported ONNX model, and the quantized ONNX model on the same set of
test poses.

Deliberately uses CPU throughout (not MPS) for rendering test images --
this workload is light, and CPU avoids the MPS instability seen during
multi-subject training.

Requires results/pose_cnn.pt, results/pose_cnn.onnx, and
results/pose_cnn_quantized.onnx.

Run:
    python src/deployment/validate_onnx.py
"""
import os
import sys

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "registration"))
from common import (
    load_subject, build_camera, sample_random_poses,
    pose_to_rt_inv, denormalize_pose, HEIGHT, WIDTH,
)
from pose_cnn import PoseCNN

from nanodrr.drr import render

N_TEST = 20
TEST_SUBJECT_ID = 1  # any subject that's already been downloaded works


def main():
    device = torch.device("cpu")  # deliberately CPU -- see module docstring

    subject, _ = load_subject(subject_id=TEST_SUBJECT_ID, device=device)
    k_inv, sdd_t = build_camera(device)

    torch.manual_seed(0)
    true_rot, true_trans = sample_random_poses(N_TEST, device)

    pt_model = PoseCNN()
    pt_model.load_state_dict(torch.load("results/pose_cnn.pt", map_location="cpu"))
    pt_model.eval()

    onnx_fp32 = ort.InferenceSession("results/pose_cnn.onnx", providers=["CPUExecutionProvider"])
    onnx_int8 = ort.InferenceSession("results/pose_cnn_quantized.onnx", providers=["CPUExecutionProvider"])
    input_name = onnx_fp32.get_inputs()[0].name

    errs = {
        "pt": {"rot": [], "trans": []},
        "onnx": {"rot": [], "trans": []},
        "int8": {"rot": [], "trans": []},
    }

    for i in range(N_TEST):
        rot_i = true_rot[i:i + 1]
        trans_i = true_trans[i:i + 1]
        rt_inv = pose_to_rt_inv(subject, rot_i, trans_i)
        with torch.no_grad():
            img = render(subject, k_inv, rt_inv, sdd_t, HEIGHT, WIDTH).sum(dim=1, keepdim=True)

        with torch.no_grad():
            pt_pred = pt_model(img)
        pt_rot, pt_trans = denormalize_pose(pt_pred)
        errs["pt"]["rot"].append((pt_rot - rot_i).abs().mean().item())
        errs["pt"]["trans"].append((pt_trans - trans_i).abs().mean().item())

        img_np = img.numpy().astype(np.float32)

        onnx_pred = torch.from_numpy(onnx_fp32.run(None, {input_name: img_np})[0])
        onnx_rot, onnx_trans = denormalize_pose(onnx_pred)
        errs["onnx"]["rot"].append((onnx_rot - rot_i).abs().mean().item())
        errs["onnx"]["trans"].append((onnx_trans - trans_i).abs().mean().item())

        int8_pred = torch.from_numpy(onnx_int8.run(None, {input_name: img_np})[0])
        int8_rot, int8_trans = denormalize_pose(int8_pred)
        errs["int8"]["rot"].append((int8_rot - rot_i).abs().mean().item())
        errs["int8"]["trans"].append((int8_trans - trans_i).abs().mean().item())

    def summarize(name, rot_errs, trans_errs):
        print(f"{name:<20} rot err {sum(rot_errs) / len(rot_errs):6.2f} deg   "
              f"trans err {sum(trans_errs) / len(trans_errs):6.2f} mm")

    print(f"\nAccuracy comparison over {N_TEST} test poses (subject {TEST_SUBJECT_ID}):\n")
    summarize("PyTorch (fp32)", errs["pt"]["rot"], errs["pt"]["trans"])
    summarize("ONNX (fp32)", errs["onnx"]["rot"], errs["onnx"]["trans"])
    summarize("ONNX (int8)", errs["int8"]["rot"], errs["int8"]["trans"])

    print("\nPyTorch and ONNX (fp32) should match closely -- export shouldn't")
    print("change predictions. INT8 quantization may show a small accuracy")
    print("drop; the deployment question is whether that drop is acceptable")
    print("given the latency/size gains measured in profile_models.py.")


if __name__ == "__main__":
    main()
