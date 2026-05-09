import argparse
import csv
import sys
import time
from pathlib import Path

# Assumes this script is run from a path where src is in PYTHONPATH,
# or we can append the parent dir to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.events_to_frames import load_events, events_to_frames, save_frames_npy, get_device

def batch_convert_events(manifest_csv: Path, output_root: Path, dt: float,
                         height: int = 768, width: int = 1024,
                         max_samples: int = None, force: bool = False,
                         device_arg: str = "auto"):
    manifest_path = manifest_csv.resolve()
    out_root_path = output_root.resolve()

    if not manifest_path.exists():
        print(f"Error: Manifest CSV {manifest_path} not found.")
        sys.exit(1)

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_samples is not None:
        rows = rows[:max_samples]

    total_videos = len(rows)
    print(f"Found {total_videos} entries to process.")

    device = get_device(device_arg)
    print(f"Batch conversion using device: {device} (Arg: {device_arg})")

    success = 0
    fails = 0
    t_start_batch = time.perf_counter()

    for i, row in enumerate(rows, start=1):
        video_id = row["video_id"]
        events_path = Path(row["events_path"])

        out_dir = out_root_path / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "event_frames.npy"

        print(f"[{i}/{total_videos}] Processing {events_path} -> {out_file}")

        t0_file = time.perf_counter()

        if out_file.exists() and not force:
            print(f"[{i}/{total_videos}] Skipping {video_id}, file already exists.")
            success += 1
            continue

        if not events_path.exists():
            print(f"[{i}/{total_videos}] Error: {events_path} not found. Skipping.")
            fails += 1
            continue

        try:
            # Timing HDF5 Read
            t0_read = time.perf_counter()
            events = load_events(events_path)
            t_read = time.perf_counter() - t0_read

            num_events = len(events)
            t_min, t_max = (events[:, 0].min(), events[:, 0].max()) if num_events > 0 else (0, 0)
            print(f"[{i}/{total_videos}] Loaded {num_events} events (t: {t_min:.1f} to {t_max:.1f}) in {t_read:.3f}s.")

            # Timing Conversion
            t0_conv = time.perf_counter()
            frames = events_to_frames(events, H=height, W=width, dt=dt, device_arg=device_arg)
            t_conv = time.perf_counter() - t0_conv

            # Timing Save
            t0_save = time.perf_counter()
            save_frames_npy(frames, out_file)
            t_save = time.perf_counter() - t0_save

            t_file = time.perf_counter() - t0_file

            print(f"[{i}/{total_videos}] Saved numpy array shape {frames.shape}, dtype {frames.dtype} in {t_file:.3f}s "
                  f"(conv: {t_conv:.3f}s, save: {t_save:.3f}s)")
            success += 1

        except Exception as e:
            print(f"[{i}/{total_videos}] Conversion crashed for {video_id}: {e}")
            fails += 1

    t_total_batch = time.perf_counter() - t_start_batch
    t_avg_batch = t_total_batch / total_videos if total_videos > 0 else 0

    print("\n====================================")
    print(f"Batch conversion complete! Time Elapsed: {t_total_batch:.3f}s (Avg: {t_avg_batch:.3f}s/sample)")
    print(f"Success: {success}, Failed: {fails}")
    print("====================================\n")

def main():
    parser = argparse.ArgumentParser(description="Batch convert events.h5 to event-frames .npy files.")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Path to manifest CSV from build_manifest.py.")
    parser.add_argument("--dt", type=float, required=True, help="Time window size for each event frame.")
    parser.add_argument("--height", type=int, default=768, help="Frame height (default: 768).")
    parser.add_argument("--width", type=int, default=1024, help="Frame width (default: 1024).")
    parser.add_argument("--output_root", type=str, required=True, help="Directory to store .npy files.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of manifest samples to process.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing output target data.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for processing.")
    args = parser.parse_args()

    batch_convert_events(
        Path(args.manifest_csv),
        Path(args.output_root),
        args.dt,
        args.height,
        args.width,
        args.max_samples,
        args.force,
        args.device
    )

if __name__ == "__main__":
    main()
