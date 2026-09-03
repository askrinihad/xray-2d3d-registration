"""
Step 2: classical 2D-3D registration by gradient descent, using nanodrr's
built-in Registration module and a normalized cross-correlation loss.

Goal: given a fixed (target) X-ray and a wrong initial pose guess, recover
the true pose by backpropagating the image similarity loss directly into
the pose parameters -- the classical (non-learned) side of 2D-3D
registration.

Run:
    python src/registration/pose_estimation.py
"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data import Subject, download_deepfluoro
from nanodrr.drr import render
from nanodrr.metrics import NormalizedCrossCorrelation2d
from nanodrr.registration import Registration


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imagepath, _ = download_deepfluoro(subject_id=1)
    subject = Subject.from_filepath(imagepath).to(device)

    sdd = 1020.0
    delx = dely = 2.0
    x0 = y0 = 0.0
    height = width = 200
    k_inv = make_k_inv(sdd, delx, dely, x0, y0, height, width, device=device)
    sdd_t = torch.tensor([sdd], device=device)

    # Ground-truth ("fixed") pose
    rt_inv_true = make_rt_inv(
        rotation=torch.tensor([[0.0, 0.0, 0.0]]),
        translation=torch.tensor([[0.0, 850.0, 0.0]]),
        orientation="AP",
        isocenter=subject.isocenter.cpu(),
    ).to(dtype=torch.float32, device=device)

    # Deliberately wrong starting ("moving") pose
    rot_init = torch.tensor([[-0.1303, 0.3461, -0.7852]]) / torch.pi * 180
    xyz_init = torch.tensor([[-11.8600, 828.8053, -24.4597]])
    rt_inv_pred = make_rt_inv(
        rotation=rot_init,
        translation=xyz_init,
        orientation="AP",
        isocenter=subject.isocenter.cpu(),
    ).to(dtype=torch.float32, device=device)

    true_img = render(subject, k_inv, rt_inv_true, sdd_t, height, width).sum(dim=1, keepdim=True)

    reg = Registration(subject, rt_inv_pred, k_inv, sdd_t, height, width)
    opt = torch.optim.Adam(
        [
            {"params": [reg._rot], "lr": 5e-2},
            {"params": [reg._xyz], "lr": 1e1},
        ],
        maximize=True,
    )
    ncc = NormalizedCrossCorrelation2d()

    losses = []
    n_iters = 500
    convergence = 0.999
    for step in tqdm(range(n_iters)):
        opt.zero_grad()
        pred = reg()
        loss = ncc(true_img, pred)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if loss > convergence:
            break

    print(f"\nConverged after {len(losses)} iterations, final NCC = {losses[-1]:.4f}")
    print("Recovered pose:", reg.pose.detach().cpu().numpy())

    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("normalized cross-correlation")
    plt.title("2D-3D registration convergence")
    plt.savefig("results/registration_convergence.png", dpi=150, bbox_inches="tight")
    print("Saved convergence plot to results/registration_convergence.png")


if __name__ == "__main__":
    main()
