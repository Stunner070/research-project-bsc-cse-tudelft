import torch
import numpy as np
from pathlib import Path

# Optional imports for privacy metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import lpips
    HAS_LPIPS = True
    # Initialize LPIPS model once at module level (AlexNet backbone)
    try:
        _lpips_model = lpips.LPIPS(net='alex', verbose=False)
    except Exception as e:
        print(f"Warning: Could not initialize LPIPS model: {e}")
        HAS_LPIPS = False
        _lpips_model = None
except ImportError:
    HAS_LPIPS = False
    _lpips_model = None

def compute_distance_matrix(features1, features2):
    """
    Compute cosine distance between two sets of features.
    features: (N, D)
    """
    f1 = torch.nn.functional.normalize(features1, p=2, dim=1)
    f2 = torch.nn.functional.normalize(features2, p=2, dim=1)
    # Cosine similarity
    sim = torch.mm(f1, f2.t())
    # Cosine distance
    dist = 1.0 - sim
    return dist

def evaluate_all_vs_all(q_features, q_labels, g_features, g_labels, q_cams=None, g_cams=None):
    """
    Evaluate Rank-1 and mAP.
    If cameras are provided, ignores self-matches (same label and same camera).
    For our baseline, we might just exclude the exact same video as query.
    Here we assume query is matched against the whole gallery minus itself.
    """
    dist_mat = compute_distance_matrix(q_features, g_features).cpu().numpy()
    q_labels = q_labels.cpu().numpy()
    g_labels = g_labels.cpu().numpy()

    if q_cams is not None and g_cams is not None:
        q_cams = q_cams.cpu().numpy()
        g_cams = g_cams.cpu().numpy()

    num_q, num_g = dist_mat.shape

    indices = np.argsort(dist_mat, axis=1)
    matches = (g_labels[indices] == q_labels[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        valid_g = np.ones(num_g, dtype=bool)
        if q_cams is not None and g_cams is not None:
            # ReID setting: same identity + same camera -> ignore
            valid_g = ~((g_labels == q_labels[q_idx]) & (g_cams == q_cams[q_idx]))
        else:
            # If no camera info, we might just exclude the exact same element if query == gallery
            # Since gallery could just be the query set itself during cross-validation:
            valid_g[q_idx] = False

        if not np.any(matches[q_idx, valid_g]):
            # No true matches in gallery
            continue

        # computing CMC and AP
        orig_indices = indices[q_idx]
        valid_indices = orig_indices[valid_g[orig_indices]]
        raw_cmc = (g_labels[valid_indices] == q_labels[q_idx]).astype(np.int32)

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:50])
        num_valid_q += 1

        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        return 0.0, 0.0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    rank1 = all_cmc[0]

    return rank1, mAP

evaluate_reid = evaluate_all_vs_all

def evaluate_query_gallery(q_features, q_labels, g_features, g_labels):
    """
    Evaluate Rank-1 and mAP for explicitly disjoint query and gallery sets.
    Returns:
        rank1 (float): Rank-1 accuracy
        mAP (float): Mean Average Precision
        top1_correct (np.ndarray): Boolean array of whether Rank-1 retrieval was correct for each query
    """
    dist_mat = compute_distance_matrix(q_features, g_features).cpu().numpy()
    q_labels = q_labels.cpu().numpy()
    g_labels = g_labels.cpu().numpy()

    num_q, num_g = dist_mat.shape

    indices = np.argsort(dist_mat, axis=1)

    all_cmc = []
    all_AP = []
    top1_correct = np.zeros(num_q, dtype=bool)
    num_valid_q = 0

    for q_idx in range(num_q):
        q_label = q_labels[q_idx]
        # Query-gallery are strictly disjoint, so all gallery samples are valid targets
        valid_indices = indices[q_idx]

        matches = (g_labels[valid_indices] == q_label).astype(np.int32)

        if not np.any(matches):
            continue

        top1_correct[q_idx] = matches[0] == 1

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:50])
        num_valid_q += 1

        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        return 0.0, 0.0, top1_correct, []

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    rank1 = all_cmc[0]

    # Compute ranks of correct identity for each query
    ranks = []
    for q_idx in range(num_q):
        q_label = q_labels[q_idx]
        valid_indices = indices[q_idx]
        matches = (g_labels[valid_indices] == q_label).astype(np.int32)

        if np.any(matches):
            correct_rank = np.argmax(matches) + 1  # ranks are 1-indexed
            ranks.append(correct_rank)
        else:
            ranks.append(len(valid_indices) + 1)  # no match found

    return rank1, mAP, top1_correct, ranks

def compute_quality_metrics(original_frames, modified_frames, device='cpu', face_detector=None):
    """
    Compute image quality metrics (SSIM, LPIPS) between original and modified frames.

    Args:
        original_frames: list/array of numpy arrays or tensors, shape (N, H, W) or (N, H, W, C) or (N, C, H, W)
        modified_frames: list/array of same shape as original_frames
        device: torch device ('cuda' or 'cpu')
        face_detector: optional face detector for alignment (not implemented in basic version)

    Returns:
        dict with keys: 'ssim_mean', 'ssim_std', 'lpips_mean', 'lpips_std',
                       'ssim_values', 'lpips_values', 'has_ssim', 'has_lpips'
    """
    results = {
        'ssim_mean': None, 'ssim_std': None, 'ssim_values': [],
        'psnr_mean': None, 'psnr_std': None, 'psnr_values': [],
        'lpips_mean': None, 'lpips_std': None, 'lpips_values': [],
        'has_ssim': HAS_SKIMAGE, 'has_psnr': HAS_SKIMAGE, 'has_lpips': HAS_LPIPS and _lpips_model is not None
    }

    if not HAS_SKIMAGE:
        print("Warning: skimage not installed, skipping SSIM computation")
    if not HAS_LPIPS or _lpips_model is None:
        print("Warning: lpips not properly initialized, skipping LPIPS computation")

    # Convert frames to tensors if needed
    if isinstance(original_frames, list):
        original_frames = np.array(original_frames)
    if isinstance(modified_frames, list):
        modified_frames = np.array(modified_frames)

    orig_tensor = torch.from_numpy(original_frames).float() if isinstance(original_frames, np.ndarray) else original_frames.float()
    mod_tensor = torch.from_numpy(modified_frames).float() if isinstance(modified_frames, np.ndarray) else modified_frames.float()

    # Normalize shape: ensure (N, C, H, W) format
    if orig_tensor.ndim == 3:  # (N, H, W)
        orig_tensor = orig_tensor.unsqueeze(1)  # (N, 1, H, W)
    if mod_tensor.ndim == 3:
        mod_tensor = mod_tensor.unsqueeze(1)

    if orig_tensor.ndim == 4 and orig_tensor.shape[1] != 1 and orig_tensor.shape[1] != 3:
        # Likely (N, H, W, C) format, permute to (N, C, H, W)
        orig_tensor = orig_tensor.permute(0, 3, 1, 2)
        mod_tensor = mod_tensor.permute(0, 3, 1, 2)

    # Ensure same spatial dimensions
    if orig_tensor.shape[2:] != mod_tensor.shape[2:]:
        h, w = orig_tensor.shape[2:]
        mod_tensor = torch.nn.functional.interpolate(mod_tensor, size=(h, w), mode='bilinear', align_corners=False)

    N = orig_tensor.shape[0]

    # Compute SSIM and PSNR
    if HAS_SKIMAGE:
        ssim_values = []
        psnr_values = []
        for i in range(N):
            orig_np = orig_tensor[i].cpu().numpy()
            mod_np = mod_tensor[i].cpu().numpy()

            # Convert to grayscale if RGB
            if orig_np.shape[0] == 3:
                orig_np = np.mean(orig_np, axis=0, keepdims=True)
            if orig_np.shape[0] == 1:
                orig_np = orig_np[0]

            if mod_np.shape[0] == 3:
                mod_np = np.mean(mod_np, axis=0)
            if mod_np.shape[0] == 1:
                mod_np = mod_np[0]

            # Normalize to [0, 1]
            orig_np = np.clip(orig_np, 0, 1)
            mod_np = np.clip(mod_np, 0, 1)

            try:
                val = ssim(orig_np, mod_np, data_range=1.0)
                ssim_values.append(float(val))
            except Exception as e:
                print(f"Warning: SSIM computation failed for frame {i}: {e}")

            try:
                val_psnr = psnr(orig_np, mod_np, data_range=1.0)
                psnr_values.append(float(val_psnr))
            except Exception as e:
                print(f"Warning: PSNR computation failed for frame {i}: {e}")

        if ssim_values:
            results['ssim_values'] = ssim_values
            results['ssim_mean'] = float(np.mean(ssim_values))
            results['ssim_std'] = float(np.std(ssim_values))
            
        if psnr_values:
            results['psnr_values'] = psnr_values
            results['psnr_mean'] = float(np.mean(psnr_values))
            results['psnr_std'] = float(np.std(psnr_values))

    # Compute LPIPS
    if HAS_LPIPS and _lpips_model is not None:
        lpips_values = []

        # Normalize to [-1, 1] for LPIPS
        orig_norm = orig_tensor.clone()
        mod_norm = mod_tensor.clone()

        if orig_norm.shape[1] == 1:
            orig_norm = orig_norm.repeat(1, 3, 1, 1)
        if mod_norm.shape[1] == 1:
            mod_norm = mod_norm.repeat(1, 3, 1, 1)

        orig_norm = (orig_norm * 2.0) - 1.0  # [0,1] -> [-1,1]
        mod_norm = (mod_norm * 2.0) - 1.0

        orig_norm = orig_norm.to(device)
        mod_norm = mod_norm.to(device)

        try:
            with torch.no_grad():
                distances = _lpips_model(orig_norm, mod_norm)
            lpips_values = distances.squeeze().cpu().numpy()
            if lpips_values.ndim == 0:
                lpips_values = [float(lpips_values)]
            else:
                lpips_values = lpips_values.tolist()

            results['lpips_values'] = lpips_values
            results['lpips_mean'] = float(np.mean(lpips_values))
            results['lpips_std'] = float(np.std(lpips_values))
        except Exception as e:
            print(f"Warning: LPIPS computation failed: {e}")

    return results
