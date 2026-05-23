import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.metrics import compute_distance_matrix

def evaluate_retrieval(embeddings_dict):
    q_emb = embeddings_dict["query"]["embeddings"]
    g_emb = embeddings_dict["gallery"]["embeddings"]

    q_ids = np.array(embeddings_dict["query"]["identity_ids"])
    g_ids = np.array(embeddings_dict["gallery"]["identity_ids"])

    if len(q_emb) == 0 or len(g_emb) == 0:
        return {
            "rank1": 0.0,
            "mAP": 0.0,
            "asr": 0.0,
            "num_queries": len(q_emb),
            "num_gallery": len(g_emb),
            "num_identities": 0
        }

    dist_mat = compute_distance_matrix(q_emb, g_emb).numpy()
    sim_mat = 1.0 - dist_mat

    num_queries = len(q_ids)
    num_gallery = len(g_ids)
    unique_ids = len(np.unique(q_ids))

    rank1_correct = 0
    aps = []

    for i in range(num_queries):
        q_id = q_ids[i]
        sims = sim_mat[i]

        ranked_indices = np.argsort(-sims)
        ranked_g_ids = g_ids[ranked_indices]

        correct_matches = (ranked_g_ids == q_id)
        if not np.any(correct_matches):
            aps.append(0.0)
            continue

        if correct_matches[0]:
            rank1_correct += 1

        correct_so_far = 0
        precisions = []
        for k, is_correct in enumerate(correct_matches):
            if is_correct:
                correct_so_far += 1
                precisions.append(correct_so_far / (k + 1.0))
        aps.append(np.mean(precisions))

    rank1 = float(rank1_correct / num_queries) if num_queries > 0 else 0.0
    mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0

    return {
        "rank1": rank1,
        "mAP": mAP,
        "asr": rank1,
        "num_queries": num_queries,
        "num_gallery": num_gallery,
        "num_identities": unique_ids
    }

