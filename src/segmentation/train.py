"""
Train the small U-Net on ARCADE for binary coronary vessel segmentation.

Run (after downloading the dataset with download_arcade.py):
    python src/segmentation/train.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ArcadeSegmentationDataset
from unet import SmallUNet


def dice_score(pred_logits, target, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    train_ds = ArcadeSegmentationDataset("data/arcade", split="train", task="syntax")
    val_ds = ArcadeSegmentationDataset("data/arcade", split="val", task="syntax")
    print(f"Train images: {len(train_ds)}  |  Val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    n_epochs = 15
    train_losses, val_dices = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch + 1}/{n_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                dices.append(dice_score(preds, masks))
        val_dices.append(sum(dices) / len(dices))

        print(f"epoch {epoch + 1}: train_loss={train_losses[-1]:.4f}  val_dice={val_dices[-1]:.4f}")

    torch.save(model.state_dict(), "results/vessel_unet.pt")
    print("Saved model to results/vessel_unet.pt")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Train loss (BCE)")
    axes[0].set_xlabel("epoch")
    axes[1].plot(val_dices)
    axes[1].set_title("Validation Dice score")
    axes[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig("results/segmentation_training.png", dpi=150, bbox_inches="tight")
    print("Saved training curves to results/segmentation_training.png")


if __name__ == "__main__":
    main()
