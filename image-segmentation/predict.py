import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from unet import UNet


def predict_mask(image_path: str, checkpoint_path: str, img_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    original = Image.open(image_path).convert("RGB")
    orig_size = original.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize(orig_size, Image.NEAREST)
    return original, mask_img


def overlay(original: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    mask_rgb = Image.new("RGB", mask.size, (255, 0, 0))
    mask_alpha = mask.point(lambda p: int(p * alpha))
    blended = original.copy()
    blended.paste(mask_rgb, (0, 0), mask_alpha)
    return blended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net segmentation on a single image")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--checkpoint", default="unet.pt")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--mask-out", default="mask.png")
    parser.add_argument("--overlay-out", default="overlay.png")
    args = parser.parse_args()

    original, mask = predict_mask(args.image, args.checkpoint, args.img_size)
    mask.save(args.mask_out)
    overlay(original, mask).save(args.overlay_out)

    print(f"Saved mask to {args.mask_out}")
    print(f"Saved overlay to {args.overlay_out}")
