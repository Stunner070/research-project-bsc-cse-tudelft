import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import math
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.models as models
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.datasets.event_video_dataset import EventVideoDataset
from src.utils.metrics import evaluate_reid, evaluate_query_gallery, compute_quality_metrics
from src.models.model_factory import build_reid_model

def get_device(device_arg: str = 'auto') -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArcFaceLoss(nn.Module):
    """Additive Angular Margin loss (ArcFace) for metric learning."""
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        # L2-normalise embeddings and weights
        x = F_nn.normalize(embeddings, p=2, dim=1)
        W = F_nn.normalize(self.W, p=2, dim=1)
        cosine = F_nn.linear(x, W)  # (B, num_classes)
        # Clamp for numerical stability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        # Add angular margin to the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = torch.cos(theta + one_hot * self.m)
        logits = logits * self.s
        return self.ce(logits, labels)

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

    # --- loss & optimizer setup ---
    arcface = None
    if backbone == 'facenet':
        arcface = ArcFaceLoss(embedding_dim=model.embed_dim, num_classes=num_classes).to(device)
        optimizer = Adam(list(model.parameters()) + list(arcface.parameters()), lr=1e-3)
    else:
        optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_mAP = -1.0
    best_val_rank1 = -1.0
    best_train_loss = float('inf')
    best_epoch = -1
    best_model_info = {}
    
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

            if arcface is not None:
                # ArcFace path: use raw embeddings
                embeddings = model.get_embedding(frames)
                loss = arcface(embeddings, labels)
                # Compute pseudo-logits for accuracy tracking
                with torch.no_grad():
                    x_n = F_nn.normalize(embeddings, p=2, dim=1)
                    W_n = F_nn.normalize(arcface.W, p=2, dim=1)
                    logits = F_nn.linear(x_n, W_n) * arcface.s
            else:
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
            val_rank1, val_mAP, _, _ = evaluate_query_gallery(val_q_features, val_q_labels, val_g_features, val_g_labels)
        else:
            val_rank1, val_mAP = 0.0, 0.0

        print(f'[{mode.upper()}] Extracting test features...')
        test_q_features, test_q_labels = extract_embeddings(model, test_query_loader, device)
        test_g_features, test_g_labels = extract_embeddings(model, test_gallery_loader, device)
        
        test_asr = 0.0
        if len(test_q_features) > 0 and len(test_g_features) > 0:
            test_rank1, test_mAP, test_top1_correct, _ = evaluate_query_gallery(test_q_features, test_q_labels, test_g_features, test_g_labels)
            
            # Evaluate ASR
            if test_query_attacked_loader is not None:
                print(f'[{mode.upper()}] Extracting attacked test features...')
                test_qa_features, test_qa_labels = extract_embeddings(model, test_query_attacked_loader, device)
                if len(test_qa_features) > 0:
                    _, _, test_attacked_top1_correct, _ = evaluate_query_gallery(test_qa_features, test_qa_labels, test_g_features, test_g_labels)
                    
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

        # Best-model selection based on validation metrics
        is_best = False
        if len(val_q_features) > 0:
            if val_mAP > best_mAP:
                is_best = True
            elif val_mAP == best_mAP:
                if val_rank1 > best_val_rank1:
                    is_best = True
                elif val_rank1 == best_val_rank1:
                    if avg_train_loss < best_train_loss:
                        is_best = True
                        
        if is_best:
            best_mAP = val_mAP
            best_val_rank1 = val_rank1
            best_train_loss = avg_train_loss
            best_epoch = epoch
            best_model_info = {
                'epoch': best_epoch,
                'val_mAP': float(val_mAP),
                'val_rank1': float(val_rank1),
                'test_mAP': float(test_mAP),
                'test_rank1': float(test_rank1),
                'checkpoint_path': str(output_path / f'{backbone}_best_{mode}.pt')
            }
            if test_query_attacked_loader is not None:
                best_model_info['test_asr'] = float(test_asr)
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, best_model_info['checkpoint_path'])

    # Save full training history
    with open(output_path / f'training_log_{mode}_{backbone}.json', 'w', encoding='utf-8') as f:
        json.dump({'mode': mode, 'best_val_mAP': float(best_mAP), 'history': logs}, f, indent=4)
        
    # Calculate averages
    if len(logs) > 0:
        avg_train_loss_all = sum(l['train_loss'] for l in logs) / len(logs)
        avg_train_acc_all = sum(l['train_acc'] for l in logs) / len(logs)
        avg_val_r1_all = sum(l['val_rank1'] for l in logs) / len(logs)
        avg_val_map_all = sum(l['val_mAP'] for l in logs) / len(logs)
        avg_test_r1_all = sum(l['test_rank1'] for l in logs) / len(logs)
        avg_test_map_all = sum(l['test_mAP'] for l in logs) / len(logs)
    else:
        avg_train_loss_all = avg_train_acc_all = avg_val_r1_all = avg_val_map_all = avg_test_r1_all = avg_test_map_all = 0.0

    last_log = logs[-1] if logs else {}

    print(f"\n==================================================")
    print(f"FINAL TRAINING SUMMARY [{mode.upper()} - {backbone}]")
    print(f"==================================================")
    print(f"Total epochs run: {epochs}")
    print(f"Best epoch (by Val mAP): {best_epoch}")
    if best_model_info:
        print(f"Best checkpoint: {best_model_info['checkpoint_path']}")
        print(f"\nBest validation metrics:")
        print(f"- Val Rank-1: {best_model_info['val_rank1']:.4f}")
        print(f"- Val mAP: {best_model_info['val_mAP']:.4f}")
        print(f"\nFinal reported test metrics from best checkpoint:")
        print(f"- Test Rank-1: {best_model_info['test_rank1']:.4f}")
        print(f"- Test mAP: {best_model_info['test_mAP']:.4f}")
        if 'test_asr' in best_model_info:
            print(f"- Test ASR: {best_model_info['test_asr']:.2f}%")
    else:
        print("No valid validation evaluation occurred.")

    if logs:
        print(f"\nLast epoch metrics:")
        print(f"- Epoch: {last_log.get('epoch', epochs)}")
        print(f"- Train Loss: {last_log.get('train_loss', 0):.4f}")
        print(f"- Train Acc: {last_log.get('train_acc', 0):.4f}")
        print(f"- Val Rank-1: {last_log.get('val_rank1', 0):.4f}")
        print(f"- Val mAP: {last_log.get('val_mAP', 0):.4f}")
        print(f"- Test Rank-1: {last_log.get('test_rank1', 0):.4f}")
        print(f"- Test mAP: {last_log.get('test_mAP', 0):.4f}")
    print(f"==================================================")

    # Save final summary JSON
    final_summary_data = {
        'mode': mode,
        'backbone': backbone,
        'total_epochs': epochs,
        'selection_metric': 'val_mAP (primary), val_rank1, train_loss',
        'best_epoch': best_epoch,
        'best_checkpoint_path': best_model_info.get('checkpoint_path', ''),
        'best_val_r1': best_model_info.get('val_rank1', 0.0),
        'best_val_map': best_model_info.get('val_mAP', 0.0),
        'best_test_r1': best_model_info.get('test_rank1', 0.0),
        'best_test_map': best_model_info.get('test_mAP', 0.0),
        'best_test_asr': best_model_info.get('test_asr', 0.0),
        'last_epoch': last_log.get('epoch', epochs),
        'last_train_loss': last_log.get('train_loss', 0.0),
        'last_train_acc': last_log.get('train_acc', 0.0),
        'last_val_r1': last_log.get('val_rank1', 0.0),
        'last_val_map': last_log.get('val_mAP', 0.0),
        'last_test_r1': last_log.get('test_rank1', 0.0),
        'last_test_map': last_log.get('test_mAP', 0.0),
        'average_metrics': {
            'train_loss': float(avg_train_loss_all),
            'train_acc': float(avg_train_acc_all),
            'val_r1': float(avg_val_r1_all),
            'val_map': float(avg_val_map_all),
            'test_r1': float(avg_test_r1_all),
            'test_map': float(avg_test_map_all)
        }
    }
    with open(output_path / f'final_summary_{mode}_{backbone}.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary_data, f, indent=4)

def run_privacy_evaluation(
    checkpoint_path: Path,
    test_csv: Path,
    backbone: str,
    mode: str,
    frames_root: Path,
    modified_query_dir: Path,
    output_dir: Path,
    device_arg: str = 'auto',
    batch_size: int = 64,
    num_workers: int = 4
) -> None:
    """
    Evaluate privacy metrics (SSIM, LPIPS, ASR) on a pre-trained model.
    """
    output_path = output_dir.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = get_device(device_arg)
    
    if backbone == 'facenet':
        transform = FaceNetTransform()
    else:
        transform = None
    
    # Load test query and gallery (clean)
    test_query_dataset = EventVideoDataset(manifest_csv=str(test_csv), mode=mode, frames_root=str(frames_root), transform=transform, role='query')
    test_gallery_dataset = EventVideoDataset(manifest_csv=str(test_csv), mode=mode, frames_root=str(frames_root), transform=transform, role='gallery')
    id_to_int = test_query_dataset.id_to_int
    
    # Load modified test queries
    modified_query_dataset = None
    try:
        modified_root = Path(modified_query_dir)
        if modified_root.exists():
            modified_query_dataset = EventVideoDataset(
                manifest_csv=str(test_csv), 
                mode=mode, 
                frames_root=str(modified_query_dir), 
                id_to_int=id_to_int,
                transform=transform, 
                role='query'
            )
    except Exception as e:
        print(f"Warning: Could not load modified test queries: {e}")
    
    test_query_loader = DataLoader(test_query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    modified_query_loader = None
    if modified_query_dataset:
        modified_query_loader = DataLoader(modified_query_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Load model
    num_classes = len(id_to_int)
    model = build_reid_model(backbone, num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    print(f"[PRIVACY_EVAL] Loaded model from {checkpoint_path}")
    
    # Extract embeddings
    print(f"[PRIVACY_EVAL] Extracting clean test features...")
    clean_q_features, clean_q_labels = extract_embeddings(model, test_query_loader, device)
    gallery_features, gallery_labels = extract_embeddings(model, test_gallery_loader, device)
    
    # Evaluate clean
    clean_rank1 = 0.0
    clean_mAP = 0.0
    if len(clean_q_features) > 0 and len(gallery_features) > 0:
        clean_rank1, clean_mAP, clean_top1_correct, _ = evaluate_query_gallery(clean_q_features, clean_q_labels, gallery_features, gallery_labels)
    
    # Evaluate modified if available
    modified_rank1 = 0.0
    modified_mAP = 0.0
    privacy_asr = 0.0
    ssim_results = None
    lpips_results = None
    
    if modified_query_loader:
        print(f"[PRIVACY_EVAL] Extracting modified test features...")
        mod_q_features, mod_q_labels = extract_embeddings(model, modified_query_loader, device)
        
        if len(mod_q_features) > 0 and len(gallery_features) > 0:
            modified_rank1, modified_mAP, modified_top1_correct, _ = evaluate_query_gallery(mod_q_features, mod_q_labels, gallery_features, gallery_labels)
            
            # Compute privacy ASR
            if len(clean_top1_correct) == len(modified_top1_correct):
                originally_correct = clean_top1_correct.sum()
                if originally_correct > 0:
                    successful_attacks = (clean_top1_correct & ~modified_top1_correct).sum()
                    privacy_asr = (successful_attacks / originally_correct) * 100.0
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"PRIVACY EVALUATION SUMMARY [{mode.upper()} - {backbone}]")
    print(f"{'='*70}")
    print(f"Model checkpoint: {checkpoint_path}\n")
    
    print(f"--- CLEAN TEST PERFORMANCE ---")
    print(f"Test Rank-1 (clean): {clean_rank1*100:.2f}%")
    print(f"Test mAP (clean):    {clean_mAP:.4f}")
    
    if modified_query_loader:
        print(f"\n--- MODIFIED TEST PERFORMANCE ---")
        print(f"Test Rank-1 (modified): {modified_rank1*100:.2f}%")
        print(f"Test mAP (modified):    {modified_mAP:.4f}")
        print(f"Privacy ASR: {privacy_asr:.2f}%")
    
    print(f"{'='*70}\n")

def print_final_summary_table(best_model_info, mode, backbone):
    """Print a clean final summary table after training."""
    if not best_model_info:
        return
    
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION SUMMARY")
    print(f"Best checkpoint from Epoch {best_model_info['epoch']}")
    print(f"{'='*70}")
    
    print(f"Model: {backbone.upper()} | Mode: {mode.upper()}")
    print(f"Best checkpoint: Epoch {best_model_info['epoch']} (Val mAP: {best_model_info['val_mAP']:.4f})")
    print(f"Checkpoint path: {best_model_info['checkpoint_path']}")
    
    print(f"\n--- Identification Metrics ---")
    print(f"Test Rank-1: {best_model_info['test_rank1']*100:.2f}%")
    print(f"Test mAP:    {best_model_info['test_mAP']:.4f}")
    
    if 'test_asr' in best_model_info and best_model_info['test_asr'] > 0:
        print(f"\n--- Attack Metrics ---")
        print(f"Test ASR (attack): {best_model_info['test_asr']:.2f}%")
    
    print(f"\n--- Image Quality Metrics ---")
    print(f"SSIM:  N/A")
    print(f"LPIPS: N/A")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    pass
