"""
Generate a synthetic image sequence simulating cardiac/respiratory motion,
by applying a smooth, time-varying non-rigid deformation to a real
coronary angiography frame (from ARCADE). The synthetic deformation field
is saved alongside each frame, giving a known ground truth to validate
tracking and motion estimation against -- real video can't provide this,
since the true motion is never actually known there.

Requires the ARCADE dataset already downloaded (see
src/segmentation/download_arcade.py).

Run:
    python src/tracking/generate_synthetic_sequence.py
"""
import glob
import os

import cv2
import numpy as np

ARCADE_ROOT = "data/arcade/arcade/syntax/train/images"
OUT_DIR = "data/synthetic_sequence"
N_FRAMES = 20
IMAGE_SIZE = 256


def load_base_frame():
    candidates = sorted(glob.glob(os.path.join(ARCADE_ROOT, "*.png")))
    if not candidates:
        raise FileNotFoundError(
            f"No images found in {ARCADE_ROOT} -- run "
            "src/segmentation/download_arcade.py first."
        )
    img = cv2.imread(candidates[0], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img


def deformation_field(t, size, amp_breathing=6.0, amp_cardiac=3.0):
    """Smooth, time-varying displacement field combining a slow
    'breathing' component and a faster 'cardiac' component -- a simple
    but genuinely non-rigid deformation, not just a rigid shift."""
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    dy = amp_breathing * np.sin(2 * np.pi * t / 20) * np.sin(np.pi * y / size)
    dx = amp_cardiac * np.sin(2 * np.pi * t / 6) * np.cos(np.pi * x / size)
    return dx.astype(np.float32), dy.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = load_base_frame()
    size = base.shape[0]
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    for t in range(N_FRAMES):
        dx, dy = deformation_field(t, size)
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        warped = cv2.remap(
            base, map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{t:03d}.png"), warped)
        np.save(os.path.join(OUT_DIR, f"flow_{t:03d}.npy"), np.stack([dx, dy], axis=-1))

    print(f"Saved {N_FRAMES} synthetic frames + ground-truth deformation fields to {OUT_DIR}")


if __name__ == "__main__":
    main()
