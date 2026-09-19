import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from model import SimpleCNN

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


def predict(image_path: str, checkpoint_path: str, top_k: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(checkpoint_path, device)

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(classes)))
    print(f"Predictions for {image_path}:")
    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
        print(f"  {classes[idx]:<8} {prob * 100:5.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--checkpoint", default="classifier.pt")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predict(args.image, args.checkpoint, args.top_k)
