import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.scripts.build_manifest import build_manifest
from src.scripts.build_splits import build_splits
from src.scripts.batch_convert_events import batch_convert_events
from src.scripts.train_compare_representations import run_training, run_privacy_evaluation
import config
import torch

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the full event-based face-ReID experiment pipeline.")
    parser.add_argument("--dt", type=float, default=config.DEFAULT_DT, help="Time window size for event-frames.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--mode", type=str, choices=["all", "event_frames_only", "dvs_only", "train_only"], default="all", help="Execution mode.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for training.")
    parser.add_argument("--backbone", type=str, choices=["resnet50", "facenet"], default="resnet50", help="Model backbone to use.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of manifest samples to process.")
    parser.add_argument("--min_clips", type=int, default=2, help="Minimum number of clips per identity required to keep it.")
    parser.add_argument("--force_conversion", action="store_true", help="Force overwrite of existing converted event arrays.")
    parser.add_argument("--frames_root_attacked", type=str, default=None, help="Path to attacked frames root for testing ASR.")
    parser.add_argument("--privacy_eval", action="store_true", help="Run privacy evaluation after training.")
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

    # Isolate training outputs by backbone
    current_models_event_frames = config.MODELS_EVENT_FRAMES / args.backbone
    current_models_dvs_avi = config.MODELS_DVS_AVI / args.backbone

    current_models_event_frames.mkdir(parents=True, exist_ok=True)
    current_models_dvs_avi.mkdir(parents=True, exist_ok=True)
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = config.MANIFEST_DIR / "manifest.csv"
    manifest_enriched = config.MANIFEST_DIR / "manifest_enriched.csv"

    print("====================================")
    print("Step 1: Building Manifest")
    print("====================================")
    build_manifest(config.V2E_ROOT, manifest_path, frames_root=config.FRAMES_ROOT)

    print("\n====================================")
    print("Step 1b: Building Enriched Canonical Splits")
    print("====================================")
    build_splits(
        manifest_path,
        None,
        manifest_enriched,
        config.SPLITS_DIR,
        id_column="identity_id",
        min_clips_per_identity=args.min_clips
    )

    train_csv = config.SPLITS_DIR / "train.csv"
    val_csv = config.SPLITS_DIR / "val.csv"
    test_csv = config.SPLITS_DIR / "test.csv"

    # Safety check
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        print("No valid split can be formed (maybe not enough data?)")
        return

    if args.mode in ["all", "event_frames_only"]:
        print("\n====================================")
        print("Step 2: Converting Events to Event-Frames")
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

        print("\n====================================")
        print("Step 3: Training on Event-Frames")
        print("====================================")
        run_training(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            mode="event_frames",
            frames_root=config.FRAMES_ROOT,
            output_dir=current_models_event_frames,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
            backbone=args.backbone,
            frames_root_attacked=Path(args.frames_root_attacked) if args.frames_root_attacked else None
        )

        # Run privacy evaluation if requested
        if args.privacy_eval and config.MODIFIED_QUERY_DIR:
            print("\n====================================")
            print("Step 3b: Running Privacy Evaluation")
            print("====================================")
            checkpoint_path = current_models_event_frames / f"{args.backbone}_best_event_frames.pt"
            if checkpoint_path.exists():
                run_privacy_evaluation(
                    checkpoint_path=checkpoint_path,
                    test_csv=test_csv,
                    backbone=args.backbone,
                    mode="event_frames",
                    frames_root=config.FRAMES_ROOT,
                    modified_query_dir=config.MODIFIED_QUERY_DIR,
                    output_dir=current_models_event_frames,
                    device_arg=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
            else:
                print(f"Warning: Could not find checkpoint at {checkpoint_path}")
                print("Skipping privacy evaluation.")
    elif args.mode == "train_only":
        print("\n====================================")
        print("Step 3: Training on Event-Frames (train_only mode)")
        print("====================================")
        run_training(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            mode="event_frames",
            frames_root=config.FRAMES_ROOT,
            output_dir=current_models_event_frames,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
            backbone=args.backbone,
            frames_root_attacked=Path(args.frames_root_attacked) if args.frames_root_attacked else None
        )

        # Run privacy evaluation if requested
        if args.privacy_eval and config.MODIFIED_QUERY_DIR:
            print("\n====================================")
            print("Step 3b: Running Privacy Evaluation")
            print("====================================")
            checkpoint_path = current_models_event_frames / f"{args.backbone}_best_event_frames.pt"
            if checkpoint_path.exists():
                run_privacy_evaluation(
                    checkpoint_path=checkpoint_path,
                    test_csv=test_csv,
                    backbone=args.backbone,
                    mode="event_frames",
                    frames_root=config.FRAMES_ROOT,
                    modified_query_dir=config.MODIFIED_QUERY_DIR,
                    output_dir=current_models_event_frames,
                    device_arg=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
            else:
                print(f"Warning: Could not find checkpoint at {checkpoint_path}")
                print("Skipping privacy evaluation.")
    else:
        print("\nSkipping Event-Frame Conversion and Training (mode is set to 'dvs_only').")

    if args.mode in ["all", "dvs_only", "train_only"]:
        print("\n====================================")
        print("Step 4: Training on dvs.avi (Grayscale)")
        print("====================================")
        run_training(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            mode="dvs_avi",
            frames_root=None,
            output_dir=current_models_dvs_avi,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
            backbone=args.backbone
        )
    else:
        print("\nSkipping dvs.avi Training (mode is set to 'event_frames_only').")

    print("\nPipeline Execution Complete.")
    print("Generated Artifacts stored remotely in:", str(config.WORK_DIR))

if __name__ == "__main__":
    main()
