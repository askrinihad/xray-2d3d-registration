"""
Step 1: generate a digitally reconstructed radiograph (DRR) from a CT
volume at a given camera pose, using nanodrr.

Unlike DiffDRR, nanodrr is pure PyTorch (no PyTorch3D dependency), so this
runs anywhere `pip install "nanodrr[all]"` works -- including plain Windows,
no CUDA toolkit or C++ build tools required.

Run:
    python src/registration/generate_drr.py
"""

import torch
import matplotlib.pyplot as plt

from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data import Subject, download_deepfluoro
from nanodrr.drr import render
from nanodrr.plot import plot_drr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Downloads the DeepFluoro CT volume the first time this runs (cached
    # after). DeepFluoro (Grupp et al.) is a real public 2D/3D X-ray
    # registration benchmark.
    imagepath, labelpath = download_deepfluoro(subject_id=1)
    subject = Subject.from_filepath(imagepath, labelpath).to(device)

    # C-arm imaging parameters
    sdd = 1020.0        # source-to-detector distance (mm)
    delx = dely = 2.0   # pixel spacing (mm)
    x0 = y0 = 0.0
    height = width = 200

    k_inv = make_k_inv(sdd, delx, dely, x0, y0, height, width, device=device)
    sdd_t = torch.tensor([sdd], device=device)

    rt_inv = make_rt_inv(
        torch.tensor([[0.0, 0.0, 0.0]], device=device),
        torch.tensor([[0.0, 850.0, 0.0]], device=device),
        orientation="AP",
        isocenter=subject.isocenter,
    )

    img = render(subject, k_inv, rt_inv, sdd_t, height, width)
    plot_drr(img.sum(dim=1, keepdim=True), ticks=False)
    plt.savefig("results/example_drr.png", dpi=150, bbox_inches="tight")
    print("Saved DRR to results/example_drr.png")


if __name__ == "__main__":
    main()
