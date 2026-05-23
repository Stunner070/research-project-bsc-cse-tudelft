import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms.functional as TF
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.facenet_reid import FaceNetReID

def extract_clip_embeddings(manifest_path, frames_root, weights_path, device, batch_size=32):
    df = pd.read_csv(manifest_path)

    # Handle column flexiblity
    id_col = "identity_id" if "identity_id" in df.columns else ("identity" if "identity" in df.columns else "video_id")

    if "role" not in df.columns:
        df = df.sort_values([id_col, "video_id"])
        roles = []
        for _, group in df.groupby(id_col):
            group_roles = ["query"] + ["gallery"] * (len(group) - 1)
            roles.extend(group_roles)
        df["role"] = roles

    df = df[df["role"].isin(["gallery", "query"])].copy()

    # Init model natively
    model = FaceNetReID(num_classes=1).to(device)
    w_path = Path(weights_path)
    if w_path.exists():
        state_dict = torch.load(w_path, map_location=device, weights_only=True)
        model.backbone.load_state_dict(state_dict, strict=False)
    model.eval()

    gallery_data = {"video_ids": [], "identity_ids": [], "embeddings": []}
    query_data = {"video_ids": [], "identity_ids": [], "embeddings": []}

    frames_batch = []
    meta_batch = []

    def process_batch(f_batch, m_batch):
        tensors = torch.cat(f_batch, dim=0).to(device)
        with torch.no_grad():
            emb = model.get_embedding(tensors).cpu()
        for i, (vid, iid, role) in enumerate(m_batch):
            if role == "gallery":
                gallery_data["video_ids"].append(vid)
                gallery_data["identity_ids"].append(iid)
                gallery_data["embeddings"].append(emb[i:i+1])
            else:
                query_data["video_ids"].append(vid)
                query_data["identity_ids"].append(iid)
                query_data["embeddings"].append(emb[i:i+1])

    for _, row in df.iterrows():
        vid = row["video_id"]
        iid = row[id_col]
        role = row["role"]

        # Only accept frames_path if it exists AND is a .npy file;
        # ignore events_path / eventspath (points to .h5, not loadable by np.load).
        if ("frames_path" in row
                and pd.notna(row["frames_path"])
                and Path(row["frames_path"]).suffix == ".npy"
                and Path(row["frames_path"]).exists()):
            npy_path = Path(row["frames_path"])
        else:
            # Default: look for the pre-generated event_frames.npy per clip
            npy_path = Path(frames_root) / vid / "event_frames.npy"

        if not npy_path.exists():
            print(f"[SKIP] {vid}: file not found at {npy_path}")
            continue

        try:
            arr = np.load(str(npy_path))
        except Exception as e:
            print(f"[SKIP] {vid}: failed to load {npy_path} ({e})")
            continue

        T = arr.shape[0]
        if T == 0:
            print(f"[SKIP] {vid}: empty array")
            continue

        frame = arr[T // 2].astype(np.float32)
        fmin, fmax = frame.min(), frame.max()
        frame = (frame - fmin) / (fmax - fmin + 1e-8)

        frame_t = torch.from_numpy(frame).unsqueeze(0) # (1, H, W)
        frame_t = TF.resize(frame_t, [160, 160], interpolation=TF.InterpolationMode.BILINEAR)
        frame_t = frame_t.repeat(3, 1, 1)
        frame_t = TF.normalize(frame_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        frames_batch.append(frame_t.unsqueeze(0))
        meta_batch.append((vid, iid, role))

        if len(frames_batch) >= batch_size:
            process_batch(frames_batch, meta_batch)
            frames_batch = []
            meta_batch = []

    if frames_batch:
        process_batch(frames_batch, meta_batch)

    if len(gallery_data["embeddings"]) > 0:
        gallery_data["embeddings"] = torch.cat(gallery_data["embeddings"], dim=0)
    else:
        gallery_data["embeddings"] = torch.empty((0, 512))

    if len(query_data["embeddings"]) > 0:
        query_data["embeddings"] = torch.cat(query_data["embeddings"], dim=0)
    else:
        query_data["embeddings"] = torch.empty((0, 512))

    return {"gallery": gallery_data, "query": query_data}

