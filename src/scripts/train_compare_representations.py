import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import torchvision.models as models

# Attempt to reach src if run from anywhere inside the project
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.datasets.event_video_dataset import EventVideoDataset

def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(
    manifest_csv: Path,
    mode: str,
    frames_root: Path | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device_arg: str = "auto"
) -> None:
    output_path = output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = EventVideoDataset(
        manifest_csv=str(manifest_csv),
        mode=mode,
        frames_root=str(frames_root) if frames_root else None
    )

    num_classes = len(dataset.id_to_int)
    print(f"[{mode.upper()}] Dataset loaded. Total samples: {len(dataset)}. Unique classes: {num_classes}")

    device = get_device(device_arg)
    print(f"Using device: {device}")

    # Random train/val split (80/20)
    indices = np.random.permutation(len(dataset))
    split = int(0.8 * len(dataset))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * frames.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(frames)

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"[{mode.upper()}] Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        }
        logs.append(log_entry)

        if val_acc >= best_val_acc and val_total > 0:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch},
                output_path / f"resnet50_best_{mode}.pt"
            )

    log_file = output_path / f"training_log_{mode}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "best_val_acc": best_val_acc,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size
            },
            "history": logs
        }, f, indent=4)

    print(f"[{mode.upper()}] Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"[{mode.upper()}] Logs and artifacts saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a ResNet on event-frames or dvs.avi to compare representations.")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Path to manifest CSV.")
    parser.add_argument("--frames_root", type=str, default=None, help="Root for event_frames.npy files.")
    parser.add_argument("--mode", type=str, choices=["event_frames", "dvs_avi"], required=True, help="Input mode.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save weights and logs.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for training.")

    args = parser.parse_args()

    run_training(
        manifest_csv=Path(args.manifest_csv),
        mode=args.mode,
        frames_root=Path(args.frames_root) if args.frames_root else None,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_arg=args.device
    )

if __name__ == "__main__":
    main()
