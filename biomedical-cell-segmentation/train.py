import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import NucleiDataset
from unet import NucleiUNet


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def evaluate(model, loader, device):
    model.eval()
    fg_dice_total, boundary_dice_total = 0.0, 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)

            fg_dice_total += dice_score(preds[:, 0:1], targets[:, 0:1]) * images.size(0)
            boundary_dice_total += dice_score(preds[:, 1:2], targets[:, 1:2]) * images.size(0)

    n = len(loader.dataset)
    return fg_dice_total / n, boundary_dice_total / n


def main():
    parser = argparse.ArgumentParser(
        description="Train a U-Net to predict cell/nucleus foreground + boundary maps "
                     "(DSB2018-style dataset layout)"
    )
    parser.add_argument("--data-dir", required=True,
                         help="Root folder containing one subfolder per sample (images/ + masks/)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--boundary-weight", type=float, default=2.0,
                         help="Extra loss weight on the boundary channel, since boundary "
                              "pixels are a small minority of the image")
    parser.add_argument("--out", default="nuclei_unet.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = NucleiDataset(args.data_dir, img_size=args.img_size)
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = NucleiUNet(in_channels=1).to(device)
    fg_criterion = nn.BCEWithLogitsLoss()
    boundary_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_score = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(images)

            fg_loss = fg_criterion(preds[:, 0:1], targets[:, 0:1])
            boundary_loss = boundary_criterion(preds[:, 1:2], targets[:, 1:2])
            loss = fg_loss + args.boundary_weight * boundary_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        fg_dice, boundary_dice = evaluate(model, val_loader, device)
        combined_score = (fg_dice + boundary_dice) / 2

        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss: {train_loss:.4f} | "
            f"fg_dice: {fg_dice:.4f} | boundary_dice: {boundary_dice:.4f}"
        )

        if combined_score > best_score:
            best_score = combined_score
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model ({best_score:.4f}) to {args.out}")

    print(f"Training done. Best combined Dice: {best_score:.4f}")


if __name__ == "__main__":
    main()
