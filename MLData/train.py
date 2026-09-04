"""
Trains a small CNN to classify rasterized spell gestures (the PNGs produced by Unity's
Tools > Spell Drawing > ML > Generate Synthetic Dataset), then exports it to ONNX so Unity
Sentis can run it at runtime.

Run:
    pip install -r requirements.txt
    python train.py
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SpellClassifier(nn.Module):
    def __init__(self, num_classes: int, image_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        reduced = image_size // 8
        assert reduced >= 1, "image_size must be at least 8 for this architecture"
        self.flat_features = 64 * reduced * reduced

        self.fc1 = nn.Linear(self.flat_features, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_dataloaders(dataset_root: Path, image_size: int, batch_size: int):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(dataset_root / "train", transform=transform)
    val_set = datasets.ImageFolder(dataset_root / "val", transform=transform)
    assert train_set.classes == val_set.classes, "train/ and val/ must contain the same class folders"

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_set.classes


def train(model, train_loader, val_loader, device, epochs: int, lr: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def export_onnx(model, image_size: int, output_path: Path, device):
    model.eval()
    dummy_input = torch.zeros(1, 1, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=15,
        external_data=False,  # keep weights in one .onnx file, not a split .onnx.data
    )
    print(f"Exported ONNX model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "Dataset",
                         help="Path to the dataset produced by the Unity generator (contains train/ and val/).")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "Output",
                         help="Where to write the trained model + classes.json.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not (args.dataset / "train").exists():
        raise SystemExit(
            f"No dataset found at {args.dataset}. Run Tools > Spell Drawing > ML > Generate Synthetic "
            f"Dataset in Unity first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from PIL import Image
    sample_path = next((args.dataset / "train").rglob("*.png"))
    image_size = Image.open(sample_path).size[0]
    print(f"Detected image size: {image_size}x{image_size}")

    train_loader, val_loader, class_names = build_dataloaders(args.dataset, image_size, args.batch_size)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}  Val samples: {len(val_loader.dataset)}")

    model = SpellClassifier(num_classes=len(class_names), image_size=image_size).to(device)
    model = train(model, train_loader, val_loader, device, args.epochs, args.lr)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "spell_classifier.pt")
    export_onnx(model, image_size, args.output / "spell_classifier.onnx", device)

    with open(args.output / "classes.json", "w") as f:
        json.dump({"classes": class_names, "image_size": image_size}, f, indent=2)

    print(f"\nDone. Copy {args.output / 'spell_classifier.onnx'} and "
          f"{args.output / 'classes.json'} into your Unity project's StreamingAssets (or wherever "
          f"you wire up Sentis) for the next step.")


if __name__ == "__main__":
    main()
