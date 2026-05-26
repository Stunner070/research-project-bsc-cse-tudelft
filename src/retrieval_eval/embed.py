"""
Retrieval embedding extraction.

Supports:
  - FaceNet (frozen, default) and InsightFace (frozen pretrained) backends
  - Optional face cropping (annotation bbox or InsightFace detector)
  - Single center-frame or multi-frame averaged embeddings

All behaviour is controlled from config.py.  Default values reproduce the
original FaceNet-center-frame baseline exactly.
"""

import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import torchvision.transforms.functional as TF
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config as cfg

# ---------------------------------------------------------------------------
# Lazy InsightFace import — only loaded when actually needed
# ---------------------------------------------------------------------------
_insightface_app = None  # cached FaceAnalysis instance


def _require_insightface():
    """Import insightface, raising a clear error if missing."""
    try:
        import insightface  # noqa: F811
        return insightface
    except ImportError as e:
        raise ImportError(
            f"Failed to import 'insightface'. (Original error: {e})\n"
            "The 'insightface' package is required when RETRIEVAL_MODEL_NAME='insightface' "
            "or RETRIEVAL_FACE_CROP_SOURCE='insightface'. Ensure you have installed it:\n"
            "  pip install insightface onnxruntime-gpu"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  1. LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_clip_array(npy_path: Path) -> Optional[np.ndarray]:
    """Load a .npy event-frame array.  Returns None on failure."""
    try:
        return np.load(str(npy_path))
    except Exception as e:
        print(f"[SKIP] Failed to load {npy_path} ({e})")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  2. FRAME SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def sample_frame_indices(
    num_frames: int,
    mode: str = "center",
    num_samples: int = 5,
    strategy: str = "uniform",
) -> List[int]:
    """Return frame indices to embed for a clip with *num_frames* frames.

    Modes
    -----
    "center"        → single index [T // 2]
    "multi_average" → *num_samples* indices chosen by *strategy*
      - "uniform"        evenly spaced across the clip
      - "center_window"  clustered around the middle
    """
    if num_frames <= 0:
        return []

    if mode == "center":
        return [num_frames // 2]

    # multi_average
    n = min(num_samples, num_frames)

    if strategy == "uniform":
        if n == 1:
            return [num_frames // 2]
        return [int(round(i * (num_frames - 1) / (n - 1))) for i in range(n)]

    if strategy == "center_window":
        center = num_frames // 2
        half = n // 2
        start = max(0, center - half)
        end = min(num_frames, start + n)
        start = max(0, end - n)  # re-clamp if we hit the upper bound
        return list(range(start, end))

    raise ValueError(f"Unknown frame sample strategy: {strategy!r}")


# ═══════════════════════════════════════════════════════════════════════════
#  3. NORMALISATION / CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def normalize_event_frame(frame: np.ndarray) -> np.ndarray:
    """Min-max normalise a single-channel frame to [0, 1] float32."""
    frame = frame.astype(np.float32)
    fmin, fmax = frame.min(), frame.max()
    return (frame - fmin) / (fmax - fmin + 1e-8)


def frame_to_uint8_rgb(frame_norm: np.ndarray) -> np.ndarray:
    """Convert a [0,1] normalised gray frame to uint8 3-channel RGB (H,W,3)."""
    gray_u8 = np.clip(frame_norm * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


# ═══════════════════════════════════════════════════════════════════════════
#  4. BBOX / FACE CROPPING
# ═══════════════════════════════════════════════════════════════════════════

def parse_h5_keys_from_row(row: pd.Series) -> Tuple[str, str, str]:
    """
    Returns (person_id, youtube_id, clip_name) from a manifest row.
    video_id format: {identity_id}{youtube_id}{clip_name}
    e.g. "id00019ITOTv0rYoM00022" -> ("id00019", "ITOTv0rYoM", "00022")
    """
    vid = str(row["video_id"])
    iid = str(row.get("identity_id", ""))
    
    if not iid or pd.isna(iid):
        iid = str(row.get("identity", ""))
        
    person_id = iid
    
    if vid.startswith(person_id):
        rest = vid[len(person_id):]
    else:
        rest = vid
        
    if len(rest) >= 5:
        clip_name = rest[-5:]
        youtube_id = rest[:-5]
    else:
        clip_name = rest
        youtube_id = ""

    return person_id, youtube_id, clip_name


def get_frame_bbox_from_h5(
    h5_file, 
    person_id: str, 
    youtube_id: str, 
    clip_name: str, 
    frame_idx: int, 
    fallback: str = "median"
) -> Optional[Tuple[float, float, float, float]]:
    """Returns normalized [x1, y1, x2, y2] from txt.h5, or None."""
    if h5_file is None:
        return None
        
    group_path = f"{person_id}/{youtube_id}/{clip_name}"
    if group_path not in h5_file:
        return None
        
    group = h5_file[group_path]
    if "frame" not in group or "bbox_xyxy_norm" not in group:
        return None
        
    frames = group["frame"][:]
    bboxes = group["bbox_xyxy_norm"][:]
    
    # Look for exact frame match
    idx_arr = np.where(frames == frame_idx)[0]
    if len(idx_arr) > 0:
        return tuple(bboxes[idx_arr[0]])
        
    # Fallback handling
    if fallback == "none":
        return None
        
    if fallback == "median":
        if "median_bbox_xyxy_norm" in group:
            return tuple(group["median_bbox_xyxy_norm"][:])
        return None
        
    if fallback == "nearest":
        if len(frames) == 0:
            return None
        nearest_idx = np.abs(frames - frame_idx).argmin()
        return tuple(bboxes[nearest_idx])
        
    return None


def normalized_bbox_to_pixel_xyxy(
    bbox_norm: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Convert normalized [x1, y1, x2, y2] to absolute pixel coordinates."""
    x1, y1, x2, y2 = bbox_norm
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def _expand_bbox(
    x1: int, y1: int, x2: int, y2: int,
    margin: float,
    img_h: int, img_w: int,
) -> Tuple[int, int, int, int]:
    """Expand bbox by *margin* fraction, clipped to image bounds."""
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * margin), int(h * margin)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy),
    )


def crop_face_from_annotation(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float,
    min_face_size: int,
) -> Optional[np.ndarray]:
    """Crop face from *frame* using an annotation bbox (x1,y1,x2,y2).

    Returns the cropped region or None if it is too small.
    *frame* may be any shape with at least 2 spatial dims (H, W, ...).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(*bbox, margin=margin, img_h=h, img_w=w)
    cw, ch = x2 - x1, y2 - y1
    if cw < min_face_size or ch < min_face_size:
        return None
    return frame[y1:y2, x1:x2]


def crop_face_with_insightface(
    frame_rgb: np.ndarray,
    app,
    margin: float,
    min_face_size: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Detect the largest face and crop it.

    Returns (cropped_rgb, bbox) or (None, None) on failure.
    *frame_rgb* must be uint8 (H, W, 3) in RGB order.
    """
    # InsightFace expects BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(frame_bgr)
    if not faces:
        return None, None

    # Pick the largest face by bbox area
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    raw_bbox = best.bbox.astype(int)  # (x1, y1, x2, y2)

    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(
        int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]),
        margin=margin, img_h=h, img_w=w,
    )
    cw, ch = x2 - x1, y2 - y1
    if cw < min_face_size or ch < min_face_size:
        return None, None

    cropped = frame_rgb[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def apply_face_crop(
    frame_norm: np.ndarray,
    frame_rgb: np.ndarray,
    row: pd.Series,
    frame_idx: int,
    insight_app=None,
    h5_file=None,
) -> Optional[np.ndarray]:
    """Dispatch face cropping based on config.  Returns cropped RGB or None.

    When RETRIEVAL_USE_FACE_CROP is False this function is never called.
    """
    source = cfg.RETRIEVAL_FACE_CROP_SOURCE
    margin = cfg.RETRIEVAL_FACE_CROP_MARGIN
    min_sz = cfg.RETRIEVAL_MIN_FACE_SIZE

    if source == "annotation":
        if h5_file is not None:
            pid, yid, cid = parse_h5_keys_from_row(row)
            bbox_norm = get_frame_bbox_from_h5(
                h5_file, pid, yid, cid, frame_idx, 
                fallback=cfg.RETRIEVAL_ANNOTATION_FALLBACK
            )
            if bbox_norm is not None:
                h, w = frame_rgb.shape[:2]
                bbox_px = normalized_bbox_to_pixel_xyxy(bbox_norm, w, h)
                return crop_face_from_annotation(frame_rgb, bbox_px, margin, min_sz)
        return None

    if source == "insightface":
        if insight_app is None:
            raise RuntimeError("InsightFace app required for crop_source='insightface'")
        cropped, _ = crop_face_with_insightface(frame_rgb, insight_app, margin, min_sz)
        return cropped

    raise ValueError(f"Unknown RETRIEVAL_FACE_CROP_SOURCE: {source!r}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. PREPROCESSING (backend-specific)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_frame_for_facenet(
    frame_norm: np.ndarray,
    crop_rgb: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Prepare a single frame for FaceNet.  Returns tensor (1, 3, 160, 160).

    If *crop_rgb* is provided (uint8 H,W,3), it is used instead of the full
    frame.  The crop is converted to grayscale-like normalised float then
    processed the same way as the full frame.
    """
    if crop_rgb is not None:
        # Convert crop to single-channel float [0,1], then process
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = frame_norm  # already [0,1] float32, (H, W)

    t = torch.from_numpy(gray).unsqueeze(0)  # (1, H, W)
    t = TF.resize(t, [160, 160], interpolation=TF.InterpolationMode.BILINEAR)
    t = t.repeat(3, 1, 1)  # (3, 160, 160)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t.unsqueeze(0)  # (1, 3, 160, 160)


def prepare_frame_for_insightface(
    frame_norm: np.ndarray,
    crop_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Prepare a single frame for InsightFace recognition.

    Returns uint8 BGR array sized (112, 112, 3) — the standard ArcFace input.
    If *crop_rgb* is provided, it is resized directly.
    """
    if crop_rgb is not None:
        img = crop_rgb
    else:
        img = frame_to_uint8_rgb(frame_norm)

    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    # InsightFace recognition expects BGR
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════
#  6. EMBEDDING BACKENDS
# ═══════════════════════════════════════════════════════════════════════════

def build_embedding_backend(device: str) -> Dict[str, Any]:
    """Construct the embedding backend based on config.

    Returns a dict with keys:
      "name"   : "facenet" | "insightface"
      "model"  : the PyTorch model (FaceNet) or None
      "app"    : the InsightFace FaceAnalysis app or None
      "device" : torch device string
      "dim"    : embedding dimensionality
    """
    name = cfg.RETRIEVAL_MODEL_NAME

    if name == "facenet":
        from src.models.facenet_reid import FaceNetReID
        model = FaceNetReID(num_classes=1).to(device)
        w_path = Path(cfg.RETRIEVAL_WEIGHTS_PATH)
        if w_path.exists():
            state_dict = torch.load(w_path, map_location=device, weights_only=True)
            model.backbone.load_state_dict(state_dict, strict=False)
        model.eval()
        return {"name": "facenet", "model": model, "app": None, "device": device, "dim": 512}

    if name == "insightface":
        insightface = _require_insightface()
        app = insightface.app.FaceAnalysis(
            name=cfg.RETRIEVAL_INSIGHTFACE_MODEL,
            providers=[cfg.RETRIEVAL_INSIGHTFACE_PROVIDER, "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if "CUDA" in cfg.RETRIEVAL_INSIGHTFACE_PROVIDER else -1,
                     det_size=cfg.RETRIEVAL_INSIGHTFACE_DET_SIZE)
        return {"name": "insightface", "model": None, "app": app, "device": device, "dim": 512}

    raise ValueError(f"Unknown RETRIEVAL_MODEL_NAME: {name!r}")


def _get_insightface_app_for_crop() -> Any:
    """Return a (cached) InsightFace FaceAnalysis app for detection-based cropping.

    Only called when RETRIEVAL_FACE_CROP_SOURCE == "insightface".
    If the main backend is already InsightFace, its app is reused in the
    caller instead — this function is a fallback for FaceNet + InsightFace-crop.
    """
    global _insightface_app
    if _insightface_app is None:
        insightface = _require_insightface()
        _insightface_app = insightface.app.FaceAnalysis(
            name=cfg.RETRIEVAL_INSIGHTFACE_MODEL,
            providers=[cfg.RETRIEVAL_INSIGHTFACE_PROVIDER, "CPUExecutionProvider"],
        )
        _insightface_app.prepare(
            ctx_id=0 if "CUDA" in cfg.RETRIEVAL_INSIGHTFACE_PROVIDER else -1,
            det_size=cfg.RETRIEVAL_INSIGHTFACE_DET_SIZE,
        )
    return _insightface_app


# ═══════════════════════════════════════════════════════════════════════════
#  7. SINGLE-FRAME EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_frame_embedding_facenet(
    tensor: torch.Tensor,
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Run FaceNet on a single preprocessed tensor (1,3,160,160).

    Returns L2-normalised 1-D numpy embedding (512,).
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        emb = model.get_embedding(tensor).cpu().numpy().flatten()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def extract_frame_embedding_insightface(
    frame_rgb: np.ndarray,
    app,
    aligned_face_bgr: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract an InsightFace embedding from a single frame.

    If *aligned_face_bgr* is provided (112×112 BGR), it is passed directly to
    the recognition model, bypassing detection.

    If not, the app detects faces in *frame_rgb* and extracts the embedding
    from the largest face.  On zero detections the full frame is resized to
    112×112 and used as a fallback (with a logged warning).

    Returns L2-normalised 1-D numpy embedding (512,) or None on total failure.
    """
    if aligned_face_bgr is not None:
        # Direct recognition on pre-aligned crop
        faces = app.get(aligned_face_bgr)
        if faces:
            emb = faces[0].embedding
        else:
            # Last resort: the recognition model in FaceAnalysis needs a
            # detected face.  Use the raw image as-is.
            # Some InsightFace versions expose rec_model directly.
            try:
                import insightface
                from insightface.utils.face_align import norm_crop
            except ImportError:
                return None
            # Treat the whole aligned image as the face
            faces = app.get(aligned_face_bgr)
            if not faces:
                return None
            emb = faces[0].embedding
    else:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(frame_bgr)
        if faces:
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = best.embedding
        else:
            # Fallback: treat full frame as pre-aligned face
            print(f"  [WARN] InsightFace detected 0 faces — using full frame as fallback")
            h, w = frame_bgr.shape[:2]
            resized = cv2.resize(frame_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
            faces = app.get(resized)
            if faces:
                emb = faces[0].embedding
            else:
                return None

    emb = emb.flatten().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ═══════════════════════════════════════════════════════════════════════════
#  8. DEBUG CROP SAVING
# ═══════════════════════════════════════════════════════════════════════════

_debug_crops_saved = 0


def save_debug_crop(
    vid: str,
    frame_idx: int,
    label: str,
    image: np.ndarray,
) -> None:
    """Save a debug crop image if enabled and under the per-run cap."""
    global _debug_crops_saved
    if not cfg.RETRIEVAL_SAVE_DEBUG_CROPS:
        return
    if _debug_crops_saved >= cfg.RETRIEVAL_DEBUG_CROP_MAX:
        return

    out_dir = Path(cfg.RETRIEVAL_DEBUG_CROP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{vid}_f{frame_idx}_{label}.png"
    out_path = out_dir / fname

    # Ensure BGR for cv2.imwrite
    if image.ndim == 3 and image.shape[2] == 3:
        save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if label != "insightface_aligned" else image
    elif image.ndim == 2:
        save_img = image
    else:
        save_img = image

    cv2.imwrite(str(out_path), save_img)
    _debug_crops_saved += 1


# ═══════════════════════════════════════════════════════════════════════════
#  9. CLIP-LEVEL EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

def extract_clip_embedding(
    arr: np.ndarray,
    backend: Dict[str, Any],
    row: Optional[pd.Series] = None,
    vid: str = "",
    h5_file = None,
) -> Optional[np.ndarray]:
    """Compute a single embedding vector for an entire clip.

    Handles frame sampling, optional cropping, per-frame embedding, and
    temporal averaging.  Returns an L2-normalised 1-D numpy array or None.
    """
    T = arr.shape[0]
    indices = sample_frame_indices(
        T,
        mode=cfg.RETRIEVAL_TEMPORAL_MODE,
        num_samples=cfg.RETRIEVAL_NUM_SAMPLE_FRAMES,
        strategy=cfg.RETRIEVAL_FRAME_SAMPLE_STRATEGY,
    )
    if not indices:
        return None

    use_crop = cfg.RETRIEVAL_USE_FACE_CROP
    backend_name = backend["name"]

    # Resolve InsightFace app for cropping (if needed and not the main backend)
    insight_app = backend["app"]
    if use_crop and cfg.RETRIEVAL_FACE_CROP_SOURCE == "insightface" and insight_app is None:
        insight_app = _get_insightface_app_for_crop()

    embeddings: List[np.ndarray] = []
    crop_fail_count = 0
    fallback_count = 0

    for idx in indices:
        frame_raw = arr[idx]
        frame_norm = normalize_event_frame(frame_raw)
        frame_rgb = frame_to_uint8_rgb(frame_norm)

        crop_rgb: Optional[np.ndarray] = None

        # --- optional cropping ---
        if use_crop:
            crop_rgb = apply_face_crop(
                frame_norm, frame_rgb, row, idx,
                insight_app=insight_app,
                h5_file=h5_file,
            )
            if crop_rgb is not None:
                save_debug_crop(vid, idx, f"{cfg.RETRIEVAL_FACE_CROP_SOURCE}_crop", crop_rgb)
            else:
                crop_fail_count += 1

        # --- compute embedding ---
        emb = None

        if backend_name == "facenet":
            tensor = prepare_frame_for_facenet(frame_norm, crop_rgb=crop_rgb)
            emb = extract_frame_embedding_facenet(tensor, backend["model"], backend["device"])

        elif backend_name == "insightface":
            if crop_rgb is not None:
                aligned = prepare_frame_for_insightface(frame_norm, crop_rgb=crop_rgb)
                emb = extract_frame_embedding_insightface(frame_rgb, backend["app"], aligned_face_bgr=aligned)
            else:
                # Let InsightFace detect in the full frame
                emb = extract_frame_embedding_insightface(frame_rgb, backend["app"])

        if emb is not None:
            embeddings.append(emb)
        else:
            save_debug_crop(vid, idx, "fullframe_fallback", frame_rgb)

    # --- fallback: if ALL crops failed, retry with full frames ---
    if use_crop and len(embeddings) == 0 and crop_fail_count > 0:
        fallback_count = len(indices)
        for idx in indices:
            frame_raw = arr[idx]
            frame_norm = normalize_event_frame(frame_raw)
            frame_rgb = frame_to_uint8_rgb(frame_norm)

            emb = None
            if backend_name == "facenet":
                tensor = prepare_frame_for_facenet(frame_norm, crop_rgb=None)
                emb = extract_frame_embedding_facenet(tensor, backend["model"], backend["device"])
            elif backend_name == "insightface":
                emb = extract_frame_embedding_insightface(frame_rgb, backend["app"])

            if emb is not None:
                embeddings.append(emb)

        if fallback_count > 0:
            print(f"  [WARN] {vid}: all {crop_fail_count} crop(s) failed — "
                  f"retried {fallback_count} frame(s) with full-frame fallback")

    if not embeddings:
        return None

    # --- aggregate ---
    if len(embeddings) == 1:
        return embeddings[0]

    stacked = np.stack(embeddings, axis=0)  # (N, D)
    mean_emb = stacked.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb


# ═══════════════════════════════════════════════════════════════════════════
#  10. PUBLIC API — extract_clip_embeddings  (main entry point)
# ═══════════════════════════════════════════════════════════════════════════

def extract_clip_embeddings(
    manifest_path,
    frames_root,
    weights_path,
    device,
    batch_size: int = 32,
    v2e_root = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract gallery and query embeddings for every clip in the manifest.

    This is the single public entry point consumed by run_retrieval_eval.py.
    The signature is intentionally unchanged from the original implementation.
    """
    global _debug_crops_saved
    _debug_crops_saved = 0  # reset per run

    df = pd.read_csv(manifest_path)

    # --- column flexibility ---
    id_col = (
        "identity_id" if "identity_id" in df.columns
        else ("identity" if "identity" in df.columns else "video_id")
    )

    if "role" not in df.columns:
        df = df.sort_values([id_col, "video_id"])
        roles = []
        for _, group in df.groupby(id_col):
            group_roles = ["query"] + ["gallery"] * (len(group) - 1)
            roles.extend(group_roles)
        df["role"] = roles

    df = df[df["role"].isin(["gallery", "query"])].copy()

    # --- h5py annotation handling ---
    h5_file = None
    use_crop = cfg.RETRIEVAL_USE_FACE_CROP
    if use_crop and cfg.RETRIEVAL_FACE_CROP_SOURCE == "annotation":
        h5_path = Path(cfg.RETRIEVAL_ANNOTATION_H5)
        if h5_path.exists():
            import h5py
            h5_file = h5py.File(h5_path, "r")
        else:
            print(f"[WARN] Annotation HDF5 not found at {h5_path}")

    # --- build backend ---
    backend = build_embedding_backend(device)
    emb_dim = backend["dim"]

    # --- logging ---
    model_label = cfg.RETRIEVAL_MODEL_NAME.upper()
    crop_label = (f"crop={cfg.RETRIEVAL_FACE_CROP_SOURCE}"
                  if cfg.RETRIEVAL_USE_FACE_CROP else "no-crop")
    temporal_label = cfg.RETRIEVAL_TEMPORAL_MODE
    if cfg.RETRIEVAL_TEMPORAL_MODE == "multi_average":
        temporal_label += (f" (n={cfg.RETRIEVAL_NUM_SAMPLE_FRAMES}, "
                           f"{cfg.RETRIEVAL_FRAME_SAMPLE_STRATEGY})")

    print(f"[EMBED] Backend: {model_label} | {crop_label} | {temporal_label}")
    if use_crop and cfg.RETRIEVAL_FACE_CROP_SOURCE == "annotation":
        print(f"[EMBED] Annotation H5: {cfg.RETRIEVAL_ANNOTATION_H5} (fallback: {cfg.RETRIEVAL_ANNOTATION_FALLBACK})")
        
    print(f"[EMBED] Manifest: {manifest_path}  ({len(df)} clips)")

    gallery_data: Dict[str, list] = {"video_ids": [], "identity_ids": [], "embeddings": []}
    query_data: Dict[str, list] = {"video_ids": [], "identity_ids": [], "embeddings": []}

    skipped = 0
    crop_fail_clips = 0
    fallback_clips = 0

    # --- fast-path: batched FaceNet with center frame and no crop ---
    use_fast_path = (
        backend["name"] == "facenet"
        and cfg.RETRIEVAL_TEMPORAL_MODE == "center"
        and not cfg.RETRIEVAL_USE_FACE_CROP
        and getattr(cfg, "RETRIEVAL_PIPELINE_MODE", "RAW_NPY") == "RAW_NPY"
    )

    if use_fast_path:
        frames_batch: List[torch.Tensor] = []
        meta_batch: List[Tuple[str, Any, str]] = []

        def _flush_batch(f_batch, m_batch):
            tensors = torch.cat(f_batch, dim=0).to(device)
            with torch.no_grad():
                emb = backend["model"].get_embedding(tensors).cpu()
            for i, (v, iid, role) in enumerate(m_batch):
                target = gallery_data if role == "gallery" else query_data
                target["video_ids"].append(v)
                target["identity_ids"].append(iid)
                target["embeddings"].append(emb[i:i+1])

        for _, row in df.iterrows():
            vid = str(row["video_id"])
            iid = row[id_col]
            role = row["role"]

            npy_path = _resolve_npy_path(row, frames_root)
            if npy_path is None or not npy_path.exists():
                print(f"[SKIP] {vid}: file not found at {npy_path}")
                skipped += 1
                continue

            arr = load_clip_array(npy_path)
            if arr is None or arr.shape[0] == 0:
                if arr is not None:
                    print(f"[SKIP] {vid}: empty array")
                skipped += 1
                continue

            frame = arr[arr.shape[0] // 2].astype(np.float32)
            fmin, fmax = frame.min(), frame.max()
            frame = (frame - fmin) / (fmax - fmin + 1e-8)

            frame_t = torch.from_numpy(frame).unsqueeze(0)
            frame_t = TF.resize(frame_t, [160, 160], interpolation=TF.InterpolationMode.BILINEAR)
            frame_t = frame_t.repeat(3, 1, 1)
            frame_t = TF.normalize(frame_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            frames_batch.append(frame_t.unsqueeze(0))
            meta_batch.append((vid, iid, role))

            if len(frames_batch) >= batch_size:
                _flush_batch(frames_batch, meta_batch)
                frames_batch = []
                meta_batch = []

        if frames_batch:
            _flush_batch(frames_batch, meta_batch)

    else:
        # --- general path: per-clip embedding ---
        for _, row in df.iterrows():
            vid = str(row["video_id"])
            iid = row[id_col]
            role = row["role"]

            pipeline_mode = getattr(cfg, "RETRIEVAL_PIPELINE_MODE", "RAW_NPY")
            arr = None

            if pipeline_mode == "E2VID":
                import e2vid_config
                if e2vid_config.E2VID_REPO_PATH not in sys.path:
                    sys.path.append(e2vid_config.E2VID_REPO_PATH)
                
                try:
                    from api import reconstruct_h5_to_memory
                except ImportError as e:
                    print(f"[ERROR] Could not import E2VID api. Make sure E2VID_REPO_PATH is correct. ({e})")
                    break

                h5_path = Path(v2e_root) / vid / "events.h5"
                
                # Check for a wrapper H5 in v2e_root
                wrapper_h5 = None
                if Path(v2e_root).exists() and Path(v2e_root).is_dir():
                    h5_files = list(Path(v2e_root).glob("*.h5"))
                    h5_files = [f for f in h5_files if f.name not in ["txt.h5", "manifest.h5", "raw_metrics.h5"]]
                    if h5_files:
                        wrapper_h5 = h5_files[0]

                # Verify H5 file existence (either individual or inside wrapper)
                exists_ok = h5_path.exists()
                if not exists_ok and wrapper_h5 is not None:
                    import h5py
                    try:
                        with h5py.File(str(wrapper_h5), "r") as f_wrap:
                            if vid in f_wrap:
                                exists_ok = True
                    except Exception:
                        pass

                if not exists_ok:
                    print(f"[SKIP] {vid}: events.h5 not found at {h5_path}")
                    skipped += 1
                    continue

                # If wrapper exists, we extract events to a temporary file
                temp_h5_path = None
                if not h5_path.exists() and wrapper_h5 is not None:
                    try:
                        events = load_events(h5_path)
                        temp_dir = Path("temp_h5")
                        temp_dir.mkdir(exist_ok=True)
                        temp_h5_path = temp_dir / f"{vid}_temp.h5"
                        import h5py
                        with h5py.File(temp_h5_path, "w") as f_temp:
                            f_temp.create_dataset("events", data=events)
                        h5_path = temp_h5_path
                    except Exception as e:
                        print(f"[SKIP] {vid}: Failed to extract events from wrapper H5 ({e})")
                        skipped += 1
                        continue

                try:
                    frames_list = reconstruct_h5_to_memory(str(h5_path), e2vid_config.E2VID_MODEL_PATH, e2vid_config.config_dict)
                    if not frames_list:
                        print(f"[SKIP] {vid}: E2VID returned empty frame list")
                        skipped += 1
                        if temp_h5_path is not None and temp_h5_path.exists():
                            temp_h5_path.unlink()
                        continue
                    arr = np.stack(frames_list)
                except Exception as e:
                    print(f"[SKIP] {vid}: E2VID reconstruction failed ({e})")
                    skipped += 1
                    if temp_h5_path is not None and temp_h5_path.exists():
                        try:
                            temp_h5_path.unlink()
                        except Exception:
                            pass
                    continue

                # Clean up temp file
                if temp_h5_path is not None and temp_h5_path.exists():
                    try:
                        temp_h5_path.unlink()
                    except Exception:
                        pass

            else:
                npy_path = _resolve_npy_path(row, frames_root)
                if npy_path is None or not npy_path.exists():
                    print(f"[SKIP] {vid}: file not found at {npy_path}")
                    skipped += 1
                    continue

                arr = load_clip_array(npy_path)
                if arr is None or arr.shape[0] == 0:
                    if arr is not None:
                        print(f"[SKIP] {vid}: empty array")
                    skipped += 1
                    continue

            emb = extract_clip_embedding(arr, backend, row=row, vid=vid, h5_file=h5_file)
            if emb is None:
                print(f"[SKIP] {vid}: embedding extraction failed")
                skipped += 1
                continue

            emb_tensor = torch.from_numpy(emb).unsqueeze(0).float()  # (1, D)
            target = gallery_data if role == "gallery" else query_data
            target["video_ids"].append(vid)
            target["identity_ids"].append(iid)
            target["embeddings"].append(emb_tensor)

    # --- aggregate embeddings ---
    for data in (gallery_data, query_data):
        if len(data["embeddings"]) > 0:
            data["embeddings"] = torch.cat(data["embeddings"], dim=0)
        else:
            data["embeddings"] = torch.empty((0, emb_dim))

    print(f"[EMBED] Done. gallery={len(gallery_data['video_ids'])}, "
          f"query={len(query_data['video_ids'])}, skipped={skipped}")

    if h5_file is not None:
        h5_file.close()

    return {"gallery": gallery_data, "query": query_data}


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_npy_path(row: pd.Series, frames_root) -> Optional[Path]:
    """Resolve the .npy path for a clip row.

    Only accepts frames_path if it exists and ends in .npy.
    Falls back to <frames_root>/<video_id>/event_frames.npy.
    Never uses events_path / eventspath (those point to .h5 files).
    """
    if ("frames_path" in row
            and pd.notna(row["frames_path"])
            and Path(row["frames_path"]).suffix == ".npy"
            and Path(row["frames_path"]).exists()):
        return Path(row["frames_path"])
    return Path(frames_root) / str(row["video_id"]) / "event_frames.npy"
