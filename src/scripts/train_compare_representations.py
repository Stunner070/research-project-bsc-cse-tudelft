import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.models as models
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.datasets.event_video_dataset import EventVideoDataset
from src.utils.metrics import evaluate_reid
def get_device(device_arg: str = 'auto') -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ReidBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            resnet = models.resnet50(weights=None)
        except TypeError:
            # Fallback for older torchvision versions on clusters like DelftBlue
            resnet = models.resnet50(pretrained=False)
        # 1-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.embed_dim = resnet.fc.in_features
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        if self.training:
            return logits, features
        else:
            return features
def extract_embeddings(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for frames, labels, _ in dataloader:
            frames = frames.to(device, non_blocking=True)
            features = model(frames)
            # if we wanted features only
            if isinstance(features, tuple):
                 features = features[1]
            all_features.append(features.cpu())
            all_labels.append(labels)
    if not all_features:
        return torch.empty(0), torch.empty(0)
    return torch.cat(all_features, 0), torch.cat(all_labels, 0)
def run_training(
    train_csv: Path,
    val_csv: Path,
    mode: str,
    frames_root: Path | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device_arg: str = 'auto'
) -> None:
    output_path = output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    train_dataset = EventVideoDataset(manifest_csv=str(train_csv), mode=mode, frames_root=str(frames_root) if frames_root else None)
    id_to_int = train_dataset.id_to_int
    num_classes = len(id_to_int)
    val_dataset = EventVideoDataset(manifest_csv=str(val_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, id_to_int=id_to_int)
    print(f'[{mode.upper()}] Datasets loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}. Train IDs: {num_classes}')
    device = get_device(device_arg)
    print(f'Using device: {device}')
    if len(train_dataset) == 0:
        print('Cannot train, train set empty.')
        return
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = ReidBaseline(num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    best_mAP = 0.0
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for frames, labels, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, features = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * frames.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        print(f'[{mode.upper()}] Extracting validation features...')
        features, labels = extract_embeddings(model, val_loader, device)
        if len(features) > 1:
            rank1, mAP = evaluate_reid(features, labels, features, labels)
        else:
            rank1, mAP = 0.0, 0.0
        print(f'[{mode.upper()}] Epoch [{epoch}/{epochs}] | Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val R1: {rank1:.4f} | Val mAP: {mAP:.4f}')
        logs.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_rank1': float(rank1),
            'val_mAP': float(mAP)
        })
        if mAP >= best_mAP and len(features) > 1:
            best_mAP = mAP
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, output_path / f'resnet50_best_{mode}.pt')
    with open(output_path / f'training_log_{mode}.json', 'w', encoding='utf-8') as f:
        json.dump({'mode': mode, 'best_val_mAP': float(best_mAP), 'history': logs}, f, indent=4)
    print(f'[{mode.upper()}] Training complete. Best mAP: {best_mAP:.4f}')
if __name__ == '__main__':
    pass
