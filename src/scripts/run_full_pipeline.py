import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.scripts.build_manifest import build_manifest
from src.scripts.batch_convert_events import batch_convert_events
from src.scripts.train_compare_representations import run_training
import config

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the full event-based face-ReID experiment pipeline.")
    parser.add_argument("--dt", type=float, default=config.DEFAULT_DT, help="Time window size for event-frames.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--mode", type=str, choices=["all", "event_frames_only", "dvs_only"], default="all", help="Execution mode.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for training.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of manifest samples to process.")
    parser.add_argument("--force_conversion", action="store_true", help="Force overwrite of existing converted event arrays.")
    args = parser.parse_args()

    # Ensure directories exist
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config.FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    config.MODELS_EVENT_FRAMES.mkdir(parents=True, exist_ok=True)
    config.MODELS_DVS_AVI.mkdir(parents=True, exist_ok=True)

    manifest_path = config.MANIFEST_DIR / "manifest.csv"

    print("====================================")
    print("Step 1: Building Manifest")
    print("====================================")
    build_manifest(config.V2E_ROOT, manifest_path)

    if args.mode in ["all", "event_frames_only"]:
        print("\n====================================")
        print("Step 2: Converting Events to Event-Frames")
        print("====================================")
        print(f"  manifest: {manifest_path}")
        print(f"  output root: {config.FRAMES_ROOT}")
        print(f"  dt: {args.dt}")
        print(f"  device: {args.device}")
        print(f"  max_samples: {args.max_samples}")
        print(f"  force: {args.force_conversion}")

        batch_convert_events(
            manifest_csv=manifest_path,
            output_root=config.FRAMES_ROOT,
            dt=args.dt,
            height=config.EVENT_HEIGHT,
            width=config.EVENT_WIDTH,
            max_samples=args.max_samples,
            force=args.force_conversion,
            device_arg=args.device
        )

        print("\n====================================")
        print("Step 3: Training on Event-Frames")
        print("====================================")
        run_training(
            manifest_csv=manifest_path,
            mode="event_frames",
            frames_root=config.FRAMES_ROOT,
            output_dir=config.MODELS_EVENT_FRAMES,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device
        )
    else:
        print("\nSkipping Event-Frame Conversion and Training (mode is set to 'dvs_only').")

    if args.mode in ["all", "dvs_only"]:
        print("\n====================================")
        print("Step 4: Training on dvs.avi (Grayscale)")
        print("====================================")
        run_training(
            manifest_csv=manifest_path,
            mode="dvs_avi",
            frames_root=None,
            output_dir=config.MODELS_DVS_AVI,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device
        )
    else:
        print("\nSkipping dvs.avi Training (mode is set to 'event_frames_only').")

    print("\nPipeline Execution Complete.")
    print("Generated Artifacts stored remotely in:", str(config.WORK_DIR))

if __name__ == "__main__":
    main()

