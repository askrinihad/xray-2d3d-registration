"""
Visualize predictions of the trained vessel segmentation model on a few
validation images -- a strong visual for the README/demo.

Run (after train.py):
    python src/segmentation/evaluate.py
"""
import torch
import matplotlib.pyplot as plt

from dataset import ArcadeSegmentationDataset
from unet import SmallUNet


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    val_ds = ArcadeSegmentationDataset("data/arcade", split="val", task="syntax")
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load("results/vessel_unet.pt", map_location=device))
    model.eval()

    n_show = 4
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    for i in range(n_show):
        img, mask = val_ds[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu()

        axes[i, 0].imshow(img[0], cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(mask[0], cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred > 0.5, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("results/segmentation_examples.png", dpi=150, bbox_inches="tight")
    print("Saved example predictions to results/segmentation_examples.png")


if __name__ == "__main__":
    main()
