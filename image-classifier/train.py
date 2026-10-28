import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import SimpleCNN

CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Folder names used by image-folder copies of CIFAR-10 (e.g. the one hosted by
# Ultralytics). Sorted alphabetically they give the same order as CLASSES.
FOLDER_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, test_tf


def get_datasets(data_dir: str, image_folder: str = None):
    """
    By default CIFAR-10 is downloaded with torchvision. If --image-folder is
    given, it is read from a folder layout instead:

        <image_folder>/train/<class_name>/*.png
        <image_folder>/test/<class_name>/*.png
    """
    train_tf, test_tf = get_transforms()

    if image_folder:
        root = Path(image_folder)
        train_set = torchvision.datasets.ImageFolder(root / "train", transform=train_tf)
        test_set = torchvision.datasets.ImageFolder(root / "test", transform=test_tf)
        if tuple(train_set.classes) != FOLDER_NAMES:
            raise ValueError(f"Unexpected class folders: {train_set.classes}")
        return train_set, test_set

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )
    return train_set, test_set


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a CNN image classifier on CIFAR-10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--image-folder", default=None,
                        help="Read CIFAR-10 from an image-folder layout instead of downloading")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", default="classifier.pt")
    parser.add_argument("--log", default="training_log.csv",
                        help="Per-epoch loss and test accuracy are written here")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set, test_set = get_datasets(args.data_dir, args.image_folder)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "test_acc", "seconds"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct = 0.0, 0
        start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_correct / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_acc: {test_acc:.4f} | {elapsed:.1f}s",
            flush=True,
        )
        with open(args.log, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(train_acc, 5),
                                    round(test_acc, 5), round(elapsed, 1)])

        # The test set is only used for reporting, never for picking a
        # checkpoint, so the saved model is simply the one after the last epoch.
        torch.save({"model_state": model.state_dict(), "classes": CLASSES}, args.out)

    print(f"Training done. Final test accuracy: {test_acc:.4f} (saved to {args.out})")


if __name__ == "__main__":
    main()
