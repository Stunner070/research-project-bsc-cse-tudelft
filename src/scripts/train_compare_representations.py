import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.models as models
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.datasets.event_video_dataset import EventVideoDataset
from src.utils.metrics import evaluate_reid, evaluate_query_gallery
from src.models.model_factory import build_reid_model

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

class FaceNetTransform:
    def __call__(self, x):
        import torch.nn.functional as F_nn
        # Resize to 160x160
        x = F_nn.interpolate(x.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False).squeeze(0)
        # If tensor is 1-channel, repeat to 3 channels along dim 0
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x

def run_training(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    mode: str,
    frames_root: Path | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device_arg: str = 'auto',
    backbone: str = 'resnet50',
    frames_root_attacked: Path | None = None
) -> None:
    output_path = output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if backbone == 'facenet':
        transform = FaceNetTransform()
    else:
        transform = None

    train_dataset = EventVideoDataset(manifest_csv=str(train_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, transform=transform)
    id_to_int = train_dataset.id_to_int
    num_classes = len(id_to_int)
    
    # Load Query and Gallery sets for Val
    val_query_dataset = EventVideoDataset(manifest_csv=str(val_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, id_to_int=id_to_int, transform=transform, role='query')
    val_gallery_dataset = EventVideoDataset(manifest_csv=str(val_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, id_to_int=id_to_int, transform=transform, role='gallery')
    
    # Load Query and Gallery sets for Test (Clean)
    test_query_dataset = EventVideoDataset(manifest_csv=str(test_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, id_to_int=id_to_int, transform=transform, role='query')
    test_gallery_dataset = EventVideoDataset(manifest_csv=str(test_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, id_to_int=id_to_int, transform=transform, role='gallery')

    # Load Attacked Test Query if provided
    test_query_attacked_dataset = None
    if frames_root_attacked is not None and mode == "event_frames":
        try:
            test_query_attacked_dataset = EventVideoDataset(manifest_csv=str(test_csv), mode=mode, frames_root=str(frames_root) if frames_root else None, frames_root_attacked=str(frames_root_attacked), is_attacked=True, id_to_int=id_to_int, transform=transform, role='query')
        except ValueError:
            print("Warning: Could not load attacked test queries.")

    print(f'[{mode.upper()}] Datasets loaded. Train: {len(train_dataset)}, Val Query: {len(val_query_dataset)}, Val Gallery: {len(val_gallery_dataset)}')
    print(f'[{mode.upper()}] Test Query: {len(test_query_dataset)}, Test Gallery: {len(test_gallery_dataset)}')
    if test_query_attacked_dataset:
        print(f'[{mode.upper()}] Test Query (Attacked): {len(test_query_attacked_dataset)}')
    print(f'[{mode.upper()}] Train IDs: {num_classes}')

    device = get_device(device_arg)

    print(f'[{mode.upper()}] Initializing {backbone} backbone...')
    model = build_reid_model(backbone, num_classes).to(device)
    print(f'[{mode.upper()}] Backbone: {backbone} | Embedding Dim: {model.embed_dim} | Input Size: {"160x160" if backbone=="facenet" else "Original"}')
    print(f'Using device: {device}')

    if len(train_dataset) == 0:
        print('Cannot train, train set empty.')
        return
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_query_loader = DataLoader(val_query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_gallery_loader = DataLoader(val_gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_query_loader = DataLoader(test_query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    test_query_attacked_loader = None
    if test_query_attacked_dataset:
         test_query_attacked_loader = DataLoader(test_query_attacked_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
        val_q_features, val_q_labels = extract_embeddings(model, val_query_loader, device)
        val_g_features, val_g_labels = extract_embeddings(model, val_gallery_loader, device)
        
        if len(val_q_features) > 0 and len(val_g_features) > 0:
            val_rank1, val_mAP, _ = evaluate_query_gallery(val_q_features, val_q_labels, val_g_features, val_g_labels)
        else:
            val_rank1, val_mAP = 0.0, 0.0

        print(f'[{mode.upper()}] Extracting test features...')
        test_q_features, test_q_labels = extract_embeddings(model, test_query_loader, device)
        test_g_features, test_g_labels = extract_embeddings(model, test_gallery_loader, device)
        
        test_asr = 0.0
        if len(test_q_features) > 0 and len(test_g_features) > 0:
            test_rank1, test_mAP, test_top1_correct = evaluate_query_gallery(test_q_features, test_q_labels, test_g_features, test_g_labels)
            
            # Evaluate ASR
            if test_query_attacked_loader is not None:
                print(f'[{mode.upper()}] Extracting attacked test features...')
                test_qa_features, test_qa_labels = extract_embeddings(model, test_query_attacked_loader, device)
                if len(test_qa_features) > 0:
                    _, _, test_attacked_top1_correct = evaluate_query_gallery(test_qa_features, test_qa_labels, test_g_features, test_g_labels)
                    
                    # ASR: % of successfully attacked queries among originally correct queries
                    # "mathematically checking clean Rank-1 != attacked Rank-1" meaning previously correct, now wrong.
                    originally_correct = test_top1_correct.sum()
                    if originally_correct > 0:
                        success_attacks = (test_top1_correct & ~test_attacked_top1_correct).sum()
                        test_asr = success_attacks / originally_correct * 100.0
        else:
            test_rank1, test_mAP = 0.0, 0.0

        if test_query_attacked_loader is not None:
            print(f'[{mode.upper()}] Epoch [{epoch}/{epochs}] | Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val R1: {val_rank1:.4f} | Val mAP: {val_mAP:.4f} | Test R1: {test_rank1:.4f} | Test mAP: {test_mAP:.4f} | Test ASR: {test_asr:.2f}%')
        else:
            print(f'[{mode.upper()}] Epoch [{epoch}/{epochs}] | Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val R1: {val_rank1:.4f} | Val mAP: {val_mAP:.4f} | Test R1: {test_rank1:.4f} | Test mAP: {test_mAP:.4f}')

        log_entry = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_rank1': float(val_rank1),
            'val_mAP': float(val_mAP),
            'test_rank1': float(test_rank1),
            'test_mAP': float(test_mAP)
        }
        if test_query_attacked_loader is not None:
            log_entry['test_asr'] = float(test_asr)
        logs.append(log_entry)

        if val_mAP >= best_mAP and len(val_q_features) > 0:
            best_mAP = val_mAP
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, output_path / f'{backbone}_best_{mode}.pt')
    with open(output_path / f'training_log_{mode}_{backbone}.json', 'w', encoding='utf-8') as f:
        json.dump({'mode': mode, 'best_val_mAP': float(best_mAP), 'history': logs}, f, indent=4)
    print(f'[{mode.upper()}] Training complete. Best mAP: {best_mAP:.4f}')

if __name__ == '__main__':
    pass
