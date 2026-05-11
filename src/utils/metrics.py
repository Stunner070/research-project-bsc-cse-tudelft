import torch
import numpy as np
from sklearn.metrics import average_precision_score

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

def evaluate_reid(q_features, q_labels, g_features, g_labels, q_cams=None, g_cams=None):
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

