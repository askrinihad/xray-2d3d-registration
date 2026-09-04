"""
Track vessel keypoints and estimate dense motion across the synthetic
sequence, then validate against the known ground-truth deformation.

- Tracking: sparse keypoints followed frame-to-frame with Lucas-Kanade
  optical flow (cv2.calcOpticalFlowPyrLK).
- Motion estimation: dense per-pixel displacement between consecutive
  frames with Farneback optical flow (cv2.calcOpticalFlowFarneback).
- Validation: End-Point Error (EPE) against the known synthetic ground
  truth -- the standard accuracy metric in the optical flow literature.

Run (after generate_synthetic_sequence.py):
    python src/tracking/track_and_estimate_motion.py
"""
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

SEQ_DIR = "data/synthetic_sequence"
RESULTS_DIR = "results"


def load_sequence():
    frame_paths = sorted(glob.glob(os.path.join(SEQ_DIR, "frame_*.png")))
    flow_paths = sorted(glob.glob(os.path.join(SEQ_DIR, "flow_*.npy")))
    frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
    fields = [np.load(p) for p in flow_paths]  # absolute displacement from frame 0
    return frames, fields


def track_keypoints(frames):
    """Sparse keypoint tracking with Lucas-Kanade optical flow."""
    p0 = cv2.goodFeaturesToTrack(frames[0], maxCorners=60, qualityLevel=0.05, minDistance=7)
    lk_params = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    tracks = [p0.reshape(-1, 2)]
    prev = frames[0]
    pts = p0
    for frame in frames[1:]:
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev, frame, pts, None, **lk_params)
        pts_next = pts_next[status.flatten() == 1]
        pts = pts_next.reshape(-1, 1, 2)
        tracks.append(pts.reshape(-1, 2))
        prev = frame
        if len(pts) < 5:
            break  # too many points lost, stop early
    return tracks


def estimate_dense_motion(frames):
    """Dense per-pixel motion estimation with Farneback optical flow,
    between each consecutive pair of frames."""
    flows = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flows.append(flow)
    return flows


def evaluate_motion_accuracy(estimated_flows, gt_fields):
    """End-Point Error (EPE): Euclidean distance between estimated and
    true per-pixel displacement, averaged per frame pair."""
    epes = []
    for i, flow in enumerate(estimated_flows):
        gt_flow = gt_fields[i + 1] - gt_fields[i]  # frame-to-frame ground truth
        error = np.sqrt(((flow - gt_flow) ** 2).sum(axis=-1))
        epes.append(error.mean())
    return epes


def main():
    frames, fields = load_sequence()
    print(f"Loaded {len(frames)} frames.")

    tracks = track_keypoints(frames)
    print(f"Tracked {len(tracks[0])} keypoints across {len(tracks)} frames "
          f"({len(tracks[-1])} still tracked at the end).")

    flows = estimate_dense_motion(frames)
    epes = evaluate_motion_accuracy(flows, fields)
    print(f"\nMean End-Point Error (EPE) across sequence: {np.mean(epes):.3f} px")
    print(f"Per-frame-pair EPE: {[round(e, 2) for e in epes]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, idx in zip(axes, [0, len(frames) // 2, len(frames) - 1]):
        ax.imshow(frames[idx], cmap="gray")
        ax.scatter(tracks[idx][:, 0], tracks[idx][:, 1], c="red", s=8)
        ax.set_title(f"Frame {idx}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tracking_keypoints.png"), dpi=150, bbox_inches="tight")

    mid = len(flows) // 2
    flow = flows[mid]
    step = 12
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step]
    fx, fy = flow[::step, ::step, 0], flow[::step, ::step, 1]

    plt.figure(figsize=(5, 5))
    plt.imshow(frames[mid], cmap="gray")
    plt.quiver(x, y, fx, fy, color="red", angles="xy", scale_units="xy", scale=0.3)
    plt.title("Estimated dense motion field (Farneback)")
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "motion_field.png"), dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(epes, marker="o")
    plt.xlabel("frame pair")
    plt.ylabel("End-Point Error (px)")
    plt.title("Motion estimation accuracy vs. known synthetic ground truth")
    plt.savefig(os.path.join(RESULTS_DIR, "motion_epe.png"), dpi=150, bbox_inches="tight")

    print(f"\nSaved visualizations to {RESULTS_DIR}/tracking_keypoints.png, "
          f"motion_field.png, motion_epe.png")


if __name__ == "__main__":
    main()
