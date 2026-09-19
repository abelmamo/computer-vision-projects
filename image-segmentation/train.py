import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split

from dataset import SegmentationDataset
from unet import UNet


def make_mask_transform(img_size: int):
    """Resize a PIL mask and convert to a binary (0/1) float tensor of shape (1, H, W)."""

    def transform(mask):
        mask = TF.resize(mask, (img_size, img_size), interpolation=TF.InterpolationMode.NEAREST)
        mask = TF.to_tensor(mask)  # (1, H, W), values in [0, 1]
        return (mask > 0.5).float()

    return transform


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def evaluate(model, loader, device):
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            total_dice += dice_coefficient(preds, masks).item() * images.size(0)
    return total_dice / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train a U-Net for binary image segmentation")
    parser.add_argument("--images-dir", required=True, help="Directory of input images")
    parser.add_argument("--masks-dir", required=True, help="Directory of ground-truth masks")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--out", default="unet.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    mask_transform = make_mask_transform(args.img_size)

    full_dataset = SegmentationDataset(
        args.images_dir, args.masks_dir,
        image_transform=image_transform, mask_transform=mask_transform,
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_dice = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | loss: {train_loss:.4f} | val_dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model ({best_dice:.4f}) to {args.out}")

    print(f"Training done. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
