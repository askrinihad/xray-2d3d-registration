"""
Shared utilities for the registration module: camera setup and random pose
sampling, used by both the training script and the classical-vs-learned
comparison.
"""
import torch
from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data import Subject, download_deepfluoro

# C-arm imaging parameters (kept identical to the classical registration demo)
SDD = 1020.0
DELX = DELY = 2.0
X0 = Y0 = 0.0
HEIGHT = WIDTH = 200

# Plausible pose ranges: small deviations around a standard AP view, similar
# to the range of C-arm repositioning seen during a real intervention,
# rather than arbitrary poses across all of SO(3).
ROT_RANGE_DEG = 20.0        # +/- degrees, each axis
TRANS_XZ_RANGE_MM = 50.0    # +/- mm, lateral/vertical offset
TRANS_Y_CENTER_MM = 850.0   # nominal source-to-isocenter distance
TRANS_Y_RANGE_MM = 50.0     # +/- mm around that nominal distance


def load_subject(subject_id=1, device="cpu"):
    imagepath, _ = download_deepfluoro(subject_id=subject_id)
    subject = Subject.from_filepath(imagepath).to(device)
    return subject, imagepath


def build_camera(device):
    k_inv = make_k_inv(SDD, DELX, DELY, X0, Y0, HEIGHT, WIDTH, device=device)
    sdd_t = torch.tensor([SDD], device=device)
    return k_inv, sdd_t


def sample_random_poses(batch_size, device):
    """Sample a batch of random, physically plausible poses (rotation in
    degrees, translation in mm)."""
    rotation = (torch.rand(batch_size, 3, device=device) * 2 - 1) * ROT_RANGE_DEG
    x = (torch.rand(batch_size, 1, device=device) * 2 - 1) * TRANS_XZ_RANGE_MM
    y = TRANS_Y_CENTER_MM + (torch.rand(batch_size, 1, device=device) * 2 - 1) * TRANS_Y_RANGE_MM
    z = (torch.rand(batch_size, 1, device=device) * 2 - 1) * TRANS_XZ_RANGE_MM
    translation = torch.cat([x, y, z], dim=1)
    return rotation, translation


def pose_to_rt_inv(subject, rotation, translation):
    return make_rt_inv(
        rotation=rotation,
        translation=translation,
        orientation="AP",
        isocenter=subject.isocenter.to(translation.device),
    ).to(dtype=torch.float32, device=rotation.device)


def normalize_pose(rotation, translation):
    """Scale pose parameters to roughly [-1, 1] for stable CNN training."""
    rot_n = rotation / ROT_RANGE_DEG
    x_n = translation[:, 0:1] / TRANS_XZ_RANGE_MM
    y_n = (translation[:, 1:2] - TRANS_Y_CENTER_MM) / TRANS_Y_RANGE_MM
    z_n = translation[:, 2:3] / TRANS_XZ_RANGE_MM
    return torch.cat([rot_n, x_n, y_n, z_n], dim=1)


def denormalize_pose(pred):
    """Inverse of normalize_pose: model output -> real rotation (deg) and
    translation (mm)."""
    rot = pred[:, 0:3] * ROT_RANGE_DEG
    x = pred[:, 3:4] * TRANS_XZ_RANGE_MM
    y = pred[:, 4:5] * TRANS_Y_RANGE_MM + TRANS_Y_CENTER_MM
    z = pred[:, 5:6] * TRANS_XZ_RANGE_MM
    return rot, torch.cat([x, y, z], dim=1)
