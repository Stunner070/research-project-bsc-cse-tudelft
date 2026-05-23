import sys
from pathlib import Path
import json
import torch
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.retrieval_eval.embed import extract_clip_embeddings
from src.retrieval_eval.retrieval_metrics import evaluate_retrieval

try:
    from src.scripts.build_manifest import build_manifest
except ImportError as e:
    raise ImportError(f"Could not import build_manifest from src.scripts.build_manifest. Error: {e}")

try:
    from src.scripts.build_splits import build_splits
except ImportError as e:
    raise ImportError(f"Could not import build_splits from src.scripts.build_splits. Error: {e}")

try:
    from src.scripts.batch_convert_events import batch_convert_events
except ImportError as e:
    raise ImportError(f"Could not import batch_convert_events from src.scripts.batch_convert_events. Error: {e}")

def ensure_prerequisites(manifest_path, frames_root, v2e_root, work_dir):
    m_path = Path(manifest_path)
    
    # Step 1 — Check for manifest_enriched.csv
    if not m_path.exists():
        raw_manifest = m_path.parent / "manifest.csv"
        if not raw_manifest.exists():
            print("[PREREQ] Building manifest.csv...")
            build_manifest(root=Path(v2e_root), output_csv=raw_manifest, frames_root=Path(frames_root))
            
        print("[PREREQ] Building manifest_enriched.csv...")
        build_splits(
            manifest_csv=raw_manifest,
            json_path=None,
            output_enriched=m_path,
            splits_dir=m_path.parent,
            id_column="identity_id",
            min_clips_per_identity=2
        )
        
    # Step 2 — Check for event frame .npy files
    df = pd.read_csv(m_path)
    if "role" in df.columns:
        df = df[df["role"].isin(["gallery", "query"])]
        
    missing_count = 0
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        
        if "frames_path" in row and pd.notna(row["frames_path"]):
            p = Path(row["frames_path"])
            if not p.exists(): missing_count += 1
        elif "events_path" in row and pd.notna(row["events_path"]):
            p = Path(row["events_path"])
            if not p.name.endswith(".npy"):  # events_path is typically .h5, so fallback to event_frames.npy
                p = Path(frames_root) / vid / "event_frames.npy"
            if not p.exists(): missing_count += 1
        else:
            p = Path(frames_root) / vid / "event_frames.npy"
            if not p.exists(): missing_count += 1
            
    if missing_count > 0:
        print(f"[PREREQ] Converting events to frames for {missing_count} missing clips...")
        batch_convert_events(
            manifest_csv=m_path,
            output_root=Path(frames_root),
            dt=config.DEFAULT_DT,
            height=config.EVENT_HEIGHT,
            width=config.EVENT_WIDTH,
            max_samples=None,
            force=False,
            device_arg="auto"
        )
    else:
        print("[PREREQ] All event frames already exist. Skipping conversion.")
        
    # Step 3 — Final check
    if not m_path.exists():
        raise RuntimeError(
            f"Prerequisite check failed! The manifest expected at {m_path} does not exist "
            f"after attempting to build it automatically. Please check your v2e_root and workspace directories."
        )

def _config_summary():
    """Build human-readable strings describing the active retrieval config."""
    model_label = config.RETRIEVAL_MODEL_NAME.upper()
    crop_label = (f"crop={config.RETRIEVAL_FACE_CROP_SOURCE}"
                  if config.RETRIEVAL_USE_FACE_CROP else "no-crop")
    temporal_label = config.RETRIEVAL_TEMPORAL_MODE
    if config.RETRIEVAL_TEMPORAL_MODE == "multi_average":
        temporal_label += (f" (n={config.RETRIEVAL_NUM_SAMPLE_FRAMES}, "
                           f"{config.RETRIEVAL_FRAME_SAMPLE_STRATEGY})")
    return model_label, crop_label, temporal_label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device auto-detected: {device}")

    model_label, crop_label, temporal_label = _config_summary()

    print(f"\n====================================")
    print(f"Retrieval Evaluation — {model_label} | {crop_label} | {temporal_label}")
    print(f"====================================")

    # --- Print dataset path info for debugging ---
    print(f"\n--- Dataset A: {config.RETRIEVAL_LABEL_A} ---")
    print(f"    V2E root:     {config.RETRIEVAL_V2E_ROOT_A}")
    print(f"    Manifest:     {config.RETRIEVAL_MANIFEST_A}")
    print(f"    Frames root:  {config.RETRIEVAL_FRAMES_ROOT_A}")
    print(f"\n--- Dataset B: {config.RETRIEVAL_LABEL_B} ---")
    print(f"    V2E root:     {config.RETRIEVAL_V2E_ROOT_B}")
    print(f"    Manifest:     {config.RETRIEVAL_MANIFEST_B}")
    print(f"    Frames root:  {config.RETRIEVAL_FRAMES_ROOT_B}")
    print()

    # Dataset A (Baseline)
    print(f"[A] Checking prerequisites for {config.RETRIEVAL_LABEL_A}...")
    ensure_prerequisites(
        manifest_path=config.RETRIEVAL_MANIFEST_A,
        frames_root=config.RETRIEVAL_FRAMES_ROOT_A,
        v2e_root=config.RETRIEVAL_V2E_ROOT_A,
        work_dir=config.RETRIEVAL_WORK_DIR_A
    )
    
    print(f"[A] Embedding {config.RETRIEVAL_LABEL_A}...")
    emb_a = extract_clip_embeddings(
        manifest_path=config.RETRIEVAL_MANIFEST_A,
        frames_root=config.RETRIEVAL_FRAMES_ROOT_A,
        weights_path=config.RETRIEVAL_WEIGHTS_PATH,
        device=device,
        batch_size=config.RETRIEVAL_BATCH_SIZE
    )
    res_a = evaluate_retrieval(emb_a)
    print(f"    done. ({res_a['num_gallery']} gallery, {res_a['num_queries']} query clips)")

    # Dataset B (Adjusted)
    print(f"\n[B] Checking prerequisites for {config.RETRIEVAL_LABEL_B}...")
    ensure_prerequisites(
        manifest_path=config.RETRIEVAL_MANIFEST_B,
        frames_root=config.RETRIEVAL_FRAMES_ROOT_B,
        v2e_root=config.RETRIEVAL_V2E_ROOT_B,
        work_dir=config.RETRIEVAL_WORK_DIR_B
    )
    
    print(f"[B] Embedding {config.RETRIEVAL_LABEL_B}...")
    emb_b = extract_clip_embeddings(
        manifest_path=config.RETRIEVAL_MANIFEST_B,
        frames_root=config.RETRIEVAL_FRAMES_ROOT_B,
        weights_path=config.RETRIEVAL_WEIGHTS_PATH,
        device=device,
        batch_size=config.RETRIEVAL_BATCH_SIZE
    )
    res_b = evaluate_retrieval(emb_b)
    print(f"    done. ({res_b['num_gallery']} gallery, {res_b['num_queries']} query clips)")

    d_rank1 = res_b["rank1"] - res_a["rank1"]
    d_map = res_b["mAP"] - res_a["mAP"]
    d_asr = res_b["asr"] - res_a["asr"]

    print("\n============================================================")
    print(f"RETRIEVAL EVALUATION — {model_label} | {crop_label} | {temporal_label}")
    print("============================================================")
    print(f"{'Metric':<10} {config.RETRIEVAL_LABEL_A:<15} {config.RETRIEVAL_LABEL_B:<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'Rank-1':<10} {res_a['rank1']:<15.4f} {res_b['rank1']:<15.4f} {d_rank1:<15.4f}")
    print(f"{'mAP':<10} {res_a['mAP']:<15.4f} {res_b['mAP']:<15.4f} {d_map:<15.4f}")
    print(f"{'ASR':<10} {res_a['asr']:<15.4f} {res_b['asr']:<15.4f} {d_asr:<15.4f}")
    print(f"{'Num Quer':<10} {res_a['num_queries']:<15} {res_b['num_queries']:<15}")
    print(f"{'Num Gall':<10} {res_a['num_gallery']:<15} {res_b['num_gallery']:<15}")
    print(f"{'Num Ids':<10} {res_a['num_identities']:<15} {res_b['num_identities']:<15}")
    print("============================================================")
    print("Delta = Adjusted minus Baseline.")
    print("Negative delta = privacy improvement (less identity leakage).")
    print("============================================================")

    if hasattr(config, 'RETRIEVAL_OUTPUT_DIR') and config.RETRIEVAL_OUTPUT_DIR is not None:
        out_dir = Path(config.RETRIEVAL_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        res_file = out_dir / "retrieval_results.json"

        out_data = {
            "config": {
                "model": config.RETRIEVAL_MODEL_NAME,
                "crop": config.RETRIEVAL_USE_FACE_CROP,
                "crop_source": config.RETRIEVAL_FACE_CROP_SOURCE,
                "temporal_mode": config.RETRIEVAL_TEMPORAL_MODE,
                "num_sample_frames": config.RETRIEVAL_NUM_SAMPLE_FRAMES,
                "frame_sample_strategy": config.RETRIEVAL_FRAME_SAMPLE_STRATEGY,
            },
            "dataset_a": {
                "label": config.RETRIEVAL_LABEL_A,
                "v2e_root": str(config.RETRIEVAL_V2E_ROOT_A),
                "manifest": str(config.RETRIEVAL_MANIFEST_A),
                "frames_root": str(config.RETRIEVAL_FRAMES_ROOT_A),
            },
            "dataset_b": {
                "label": config.RETRIEVAL_LABEL_B,
                "v2e_root": str(config.RETRIEVAL_V2E_ROOT_B),
                "manifest": str(config.RETRIEVAL_MANIFEST_B),
                "frames_root": str(config.RETRIEVAL_FRAMES_ROOT_B),
            },
            "results_a": res_a,
            "results_b": res_b,
            "delta": {
                "rank1": d_rank1,
                "mAP": d_map,
                "asr": d_asr
            }
        }

        with open(res_file, "w") as f:
            json.dump(out_data, f, indent=4)
        print(f"\nSaved results to {res_file}")

if __name__ == "__main__":
    main()
