import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import NucleiDataset, find_samples
from unet import NucleiUNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        description="Train a U-Net to predict nucleus foreground + boundary maps"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Training data, in either layout supported by dataset.py")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--boundary-weight", type=float, default=2.0,
                        help="Extra loss weight on the boundary channel, since boundary "
                             "pixels are a small minority of the image")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable random flips and 90-degree rotations during training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", default="nuclei_unet.pt")
    parser.add_argument("--log", default="training_log.csv",
                        help="Per-epoch loss and validation Dice are written here")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed, seeded train/validation split so runs are reproducible.
    n_samples = len(find_samples(args.data_dir))
    order = np.random.RandomState(args.seed).permutation(n_samples)
    val_size = max(1, int(n_samples * args.val_split))
    val_idx, train_idx = order[:val_size].tolist(), order[val_size:].tolist()

    train_set = NucleiDataset(args.data_dir, args.img_size,
                              augment=not args.no_augment, indices=train_idx)
    val_set = NucleiDataset(args.data_dir, args.img_size, augment=False, indices=val_idx)
    print(f"Train samples: {len(train_set)} | validation samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = NucleiUNet(in_channels=1).to(device)
    fg_criterion = nn.BCEWithLogitsLoss()
    boundary_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_fg_dice", "val_boundary_dice", "seconds"])

    best_score = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()

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

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        fg_dice, boundary_dice = evaluate(model, val_loader, device)
        combined_score = (fg_dice + boundary_dice) / 2
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss: {train_loss:.4f} | "
            f"fg_dice: {fg_dice:.4f} | boundary_dice: {boundary_dice:.4f} | {elapsed:.0f}s",
            flush=True,
        )
        with open(args.log, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(fg_dice, 5),
                                    round(boundary_dice, 5), round(elapsed, 1)])

        if combined_score > best_score:
            best_score = combined_score
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model ({best_score:.4f}) to {args.out}", flush=True)

    print(f"Training done. Best combined Dice: {best_score:.4f}")


if __name__ == "__main__":
    main()
