import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.scripts.build_manifest import build_manifest
from src.scripts.build_splits import build_splits
from src.scripts.batch_convert_events import batch_convert_events
import config
import torch

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the event-based data preparation pipeline.")
    parser.add_argument("--dt", type=float, default=config.DEFAULT_DT, help="Time window size for event-frames.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for conversion.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of manifest samples to process.")
    parser.add_argument("--min_clips", type=int, default=2, help="Minimum number of clips per identity required to keep it.")
    parser.add_argument("--force_conversion", action="store_true", help="Force overwrite of existing converted event arrays.")
    args = parser.parse_args()

    # GPU Check
    print("====================================")
    print("System Check: Computing Devices")
    print("====================================")
    has_gpu = torch.cuda.is_available()
    
    if args.device == "auto":
        resolved_device = "cuda" if has_gpu else "cpu"
    else:
        resolved_device = args.device

    if has_gpu and resolved_device == "cuda":
        print(f"-> GPU detected! ({torch.cuda.get_device_name(0)})")
        print("-> The pipeline will run accelerated on the GPU.")
    elif resolved_device == "cpu":
        print("-> GPU not detected or CPU forced.")
        print("-> The pipeline will run on standard CPU.")
        
    args.device = resolved_device

    # Ensure directories exist
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config.FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = config.MANIFEST_DIR / "manifest.csv"
    manifest_enriched = config.MANIFEST_DIR / "manifest_enriched.csv"

    print("====================================")
    print("Step 1: Building Manifest")
    print("====================================")
    build_manifest(config.V2E_ROOT, manifest_path, frames_root=config.FRAMES_ROOT)

    print("\n====================================")
    print("Step 2: Building Enriched Canonical Splits")
    print("====================================")
    build_splits(
        manifest_path,
        None,
        manifest_enriched,
        config.SPLITS_DIR,
        id_column="identity_id",
        min_clips_per_identity=args.min_clips
    )

    print("\n====================================")
    print("Step 3: Converting Events to Event-Frames")
    print("====================================")
    print(f"  manifest: {manifest_enriched}")
    print(f"  output root: {config.FRAMES_ROOT}")
    print(f"  dt: {args.dt}")
    print(f"  device: {args.device}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  force: {args.force_conversion}")

    batch_convert_events(
        manifest_csv=manifest_enriched,
        output_root=config.FRAMES_ROOT,
        dt=args.dt,
        height=config.EVENT_HEIGHT,
        width=config.EVENT_WIDTH,
        max_samples=args.max_samples,
        force=args.force_conversion,
        device_arg=args.device
    )

    print("\nData Preparation Complete.")
    print("Generated Artifacts stored remotely in:", str(config.WORK_DIR))

if __name__ == "__main__":
    main()
