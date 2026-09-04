"""
Train PoseCNN to predict a 6-DoF pose directly from a single DRR, using
synthetic (image, pose) pairs generated on the fly with nanodrr.

Trains across MULTIPLE DeepFluoro subjects and holds one out entirely, so
generalization to an unseen patient can actually be measured in
compare_classical_vs_learned.py -- rather than only demonstrating the
technique on a single, likely-overfit subject.

DeepFluoro provides 6 cadaveric subjects total (Grupp et al., 2020).

Memory note: this preloads all training subjects' CT volumes onto the
device at once. If you hit an out-of-memory error, reduce
TRAIN_SUBJECT_IDS to fewer subjects (e.g. [1, 2, 3]).

Run:
    python src/registration/train_pose_cnn.py
"""
import os
import random
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from nanodrr.drr import render

from common import (
    load_subject, build_camera, sample_random_poses,
    pose_to_rt_inv, normalize_pose, HEIGHT, WIDTH,
)
from pose_cnn import PoseCNN

TRAIN_SUBJECT_IDS = [1, 2, 3]  # reduced from 5 -- holding many CT volumes at
                                 # once on Apple's MPS GPU was triggering
                                 # driver-level "command buffer" errors during
                                 # testing; fewer subjects reduces memory
                                 # pressure. Bump back up if it stays stable.
TEST_SUBJECT_ID = 6  # held out entirely -- never seen during training


def load_subject_with_retry(sid, device, max_attempts=4):
    for attempt in range(1, max_attempts + 1):
        try:
            subject, _ = load_subject(subject_id=sid, device=device)
            return subject
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = 5 * attempt
            print(f"  download of subject {sid} failed ({e}); retrying in {wait}s "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)


def main():
    # If MPS keeps throwing "command buffer" errors even with fewer
    # subjects, force CPU instead (slower, but numerically trustworthy):
    #   device = torch.device("cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)
    print(f"Training subjects: {TRAIN_SUBJECT_IDS}  |  held-out test subject: {TEST_SUBJECT_ID}")

    print("Loading training subjects (downloads each CT volume once, cached after)...")
    subjects = {}
    for sid in TRAIN_SUBJECT_IDS:
        subjects[sid] = load_subject_with_retry(sid, device)
        print(f"  loaded subject {sid}")

    k_inv, sdd_t = build_camera(device)

    model = PoseCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    batch_size = 16
    n_steps = 600  # more than the single-subject version, since the model
                   # now has more anatomical variation to learn across
    losses = []

    for step in tqdm(range(n_steps)):
        subject = subjects[random.choice(TRAIN_SUBJECT_IDS)]

        rotation, translation = sample_random_poses(batch_size, device)
        rt_inv = pose_to_rt_inv(subject, rotation, translation)
        with torch.no_grad():
            imgs = render(subject, k_inv, rt_inv, sdd_t, HEIGHT, WIDTH).sum(dim=1, keepdim=True)

        target = normalize_pose(rotation, translation)
        pred = model(imgs)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if torch.isnan(loss) or (step > 20 and loss.item() < 1e-6):
            print(f"\nWARNING: suspicious loss ({loss.item()}) at step {step} -- "
                  "this usually means the GPU silently dropped a computation, "
                  "not real convergence. Consider switching device to CPU above.")

        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.5f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/pose_cnn.pt")
    print("Saved trained model to results/pose_cnn.pt")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("MSE loss (normalized pose)")
    plt.title(f"Learned pose-prediction training curve (subjects {TRAIN_SUBJECT_IDS})")
    plt.savefig("results/pose_cnn_training.png", dpi=150, bbox_inches="tight")
    print("Saved training curve to results/pose_cnn_training.png")


if __name__ == "__main__":
    main()