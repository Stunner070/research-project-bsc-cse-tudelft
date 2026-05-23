import sys
from pathlib import Path
import json
import torch
import numpy as np
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


def align_manifests(
    manifest_a: Path,
    manifest_b: Path,
    min_clips_per_identity: int = 2,
    seed: int = 42,
) -> tuple:
    """Align two manifests so they share the exact same identities and video IDs.

    Steps:
      1. Load both enriched manifests.
      2. Intersect on identity_id — keep only identities present in BOTH.
      3. Intersect on video_id  — keep only clips present in BOTH.
      4. Re-apply min_clips_per_identity filter on the intersection.
      5. Assign query/gallery roles consistently (same query clip for each
         identity in both A and B) using a fixed random seed.
      6. Overwrite both manifest files with the aligned versions.

    Returns (df_a, df_b) after alignment.
    """
    df_a = pd.read_csv(manifest_a)
    df_b = pd.read_csv(manifest_b)

    # Determine the identity column
    id_col = "identity_id" if "identity_id" in df_a.columns else (
        "identity" if "identity" in df_a.columns else "video_id"
    )

    print(f"\n[ALIGN] Before alignment:")
    print(f"  A: {len(df_a)} clips, {df_a[id_col].nunique()} identities")
    print(f"  B: {len(df_b)} clips, {df_b[id_col].nunique()} identities")

    # Step 1: intersect identities
    common_ids = set(df_a[id_col].unique()) & set(df_b[id_col].unique())
    df_a = df_a[df_a[id_col].isin(common_ids)].copy()
    df_b = df_b[df_b[id_col].isin(common_ids)].copy()

    # Step 2: intersect video IDs
    common_vids = set(df_a["video_id"].unique()) & set(df_b["video_id"].unique())
    df_a = df_a[df_a["video_id"].isin(common_vids)].copy()
    df_b = df_b[df_b["video_id"].isin(common_vids)].copy()

    # Step 3: re-apply minimum clips filter on the intersection
    counts = df_a.groupby(id_col)["video_id"].count()
    valid_ids = counts[counts >= min_clips_per_identity].index.tolist()
    df_a = df_a[df_a[id_col].isin(valid_ids)].copy()
    df_b = df_b[df_b[id_col].isin(valid_ids)].copy()

    # Step 4: assign consistent roles across both datasets
    # Sort identically so row order matches
    df_a = df_a.sort_values([id_col, "video_id"]).reset_index(drop=True)
    df_b = df_b.sort_values([id_col, "video_id"]).reset_index(drop=True)

    np.random.seed(seed)
    roles = []
    for _, group in df_a.groupby(id_col):
        n = len(group)
        # Pick one random query index within this identity's clips
        query_local = np.random.randint(0, n)
        group_roles = ["gallery"] * n
        group_roles[query_local] = "query"
        roles.extend(group_roles)

    df_a["role"] = roles
    df_b["role"] = roles  # same roles for both — same clips, same order

    # Also ensure the 'identity' column exists (used by embed.py)
    if "identity" not in df_a.columns:
        df_a["identity"] = df_a[id_col]
    if "identity" not in df_b.columns:
        df_b["identity"] = df_b[id_col]

    print(f"[ALIGN] After alignment:")
    print(f"  Common identities: {len(valid_ids)}")
    print(f"  Common clips:      {len(df_a)}")
    print(f"  Queries per set:   {(df_a['role'] == 'query').sum()}")
    print(f"  Gallery per set:   {(df_a['role'] == 'gallery').sum()}")

    # Step 5: overwrite the manifest files
    manifest_a.parent.mkdir(parents=True, exist_ok=True)
    manifest_b.parent.mkdir(parents=True, exist_ok=True)
    df_a.to_csv(manifest_a, index=False)
    df_b.to_csv(manifest_b, index=False)
    print(f"[ALIGN] Wrote aligned manifests:")
    print(f"  A: {manifest_a}")
    print(f"  B: {manifest_b}")

    return df_a, df_b

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

    # Dataset A (Baseline) — ensure prerequisites
    print(f"[A] Checking prerequisites for {config.RETRIEVAL_LABEL_A}...")
    ensure_prerequisites(
        manifest_path=config.RETRIEVAL_MANIFEST_A,
        frames_root=config.RETRIEVAL_FRAMES_ROOT_A,
        v2e_root=config.RETRIEVAL_V2E_ROOT_A,
        work_dir=config.RETRIEVAL_WORK_DIR_A
    )

    # Dataset B (Adjusted) — ensure prerequisites
    print(f"\n[B] Checking prerequisites for {config.RETRIEVAL_LABEL_B}...")
    ensure_prerequisites(
        manifest_path=config.RETRIEVAL_MANIFEST_B,
        frames_root=config.RETRIEVAL_FRAMES_ROOT_B,
        v2e_root=config.RETRIEVAL_V2E_ROOT_B,
        work_dir=config.RETRIEVAL_WORK_DIR_B
    )

    # Align manifests — intersect identities & clips, assign consistent roles
    align_manifests(
        manifest_a=Path(config.RETRIEVAL_MANIFEST_A),
        manifest_b=Path(config.RETRIEVAL_MANIFEST_B),
        min_clips_per_identity=2,
        seed=42,
    )

    if getattr(config, "RETRIEVAL_USE_FACE_MODELS", True):
        # Embed Dataset A
        print(f"\n[A] Embedding {config.RETRIEVAL_LABEL_A}...")
        emb_a = extract_clip_embeddings(
            manifest_path=config.RETRIEVAL_MANIFEST_A,
            frames_root=config.RETRIEVAL_FRAMES_ROOT_A,
            weights_path=config.RETRIEVAL_WEIGHTS_PATH,
            device=device,
            batch_size=config.RETRIEVAL_BATCH_SIZE
        )
        res_a = evaluate_retrieval(emb_a)
        print(f"    done. ({res_a['num_gallery']} gallery, {res_a['num_queries']} query clips)")

        # Embed Dataset B
        print(f"\n[B] Embedding {config.RETRIEVAL_LABEL_B}...")
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

    else:
        print("\n[EVAL] Running intrinsic structural metrics (SSIM, PSNR) mode...")
        from src.utils.metrics import compute_quality_metrics
        from src.retrieval_eval.embed import load_clip_array, normalize_event_frame, _resolve_npy_path, sample_frame_indices

        df_a = pd.read_csv(config.RETRIEVAL_MANIFEST_A)
        df_b = pd.read_csv(config.RETRIEVAL_MANIFEST_B)

        ssim_list = []
        psnr_list = []

        print(f"Processing {len(df_a)} matched clips for structural evaluation...")
        for i, row_a in df_a.iterrows():
            vid = row_a["video_id"]
            # df_b is exactly aligned, but we query by vid to be safe
            matching_b = df_b[df_b["video_id"] == vid]
            if len(matching_b) == 0:
                continue
            row_b = matching_b.iloc[0]
            
            path_a = _resolve_npy_path(row_a, config.RETRIEVAL_FRAMES_ROOT_A)
            path_b = _resolve_npy_path(row_b, config.RETRIEVAL_FRAMES_ROOT_B)
            
            if path_a is None or path_b is None or not path_a.exists() or not path_b.exists():
                continue
                
            arr_a = load_clip_array(path_a)
            arr_b = load_clip_array(path_b)
            if arr_a is None or arr_b is None: continue
            
            T = min(len(arr_a), len(arr_b))
            indices = sample_frame_indices(
                T, 
                mode=config.RETRIEVAL_TEMPORAL_MODE, 
                num_samples=config.RETRIEVAL_NUM_SAMPLE_FRAMES, 
                strategy=config.RETRIEVAL_FRAME_SAMPLE_STRATEGY
            )
            
            frames_a = []
            frames_b = []
            for idx in indices:
                frames_a.append(normalize_event_frame(arr_a[idx]))
                frames_b.append(normalize_event_frame(arr_b[idx]))
                
            metrics = compute_quality_metrics(frames_a, frames_b, device=device)
            if metrics.get("ssim_values"):
                ssim_list.extend(metrics["ssim_values"])
            if metrics.get("psnr_values"):
                psnr_list.extend(metrics["psnr_values"])

        print("\n============================================================")
        print(f"STRUCTURAL EVALUATION — {temporal_label}")
        print("============================================================")
        if ssim_list:
            print(f"{'Mean SSIM:':<15} {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
        else:
            print(f"{'Mean SSIM:':<15} N/A")
            
        if psnr_list:
            print(f"{'Mean PSNR:':<15} {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}")
        else:
            print(f"{'Mean PSNR:':<15} N/A")
        print(f"{'Total Frames:':<15} {len(ssim_list)}")
        print("============================================================")

if __name__ == "__main__":
    main()
