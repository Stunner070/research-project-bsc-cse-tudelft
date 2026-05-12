import argparse
import csv
import sys
import time
from pathlib import Path
import concurrent.futures

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

    def process_row(args):
        i, row = args
        video_id = row["video_id"]
        events_path = Path(row["events_path"])

        out_dir = out_root_path / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "event_frames.npy"

        if out_file.exists() and not force:
            return (True, f"[{i}/{total_videos}] Skipping {video_id}, file already exists.")

        if not events_path.exists():
            return (False, f"[{i}/{total_videos}] Error: {events_path} not found. Skipping.")

        try:
            t0_file = time.perf_counter()

            # Timing HDF5 Read
            t0_read = time.perf_counter()
            events = load_events(events_path)
            t_read = time.perf_counter() - t0_read

            num_events = len(events)

            # Timing Conversion
            t0_conv = time.perf_counter()
            frames = events_to_frames(events, H=height, W=width, dt=dt, device_arg=device_arg)
            t_conv = time.perf_counter() - t0_conv

            # Timing Save
            t0_save = time.perf_counter()
            save_frames_npy(frames, out_file)
            t_save = time.perf_counter() - t0_save

            t_file = time.perf_counter() - t0_file

            msg = f"[{i}/{total_videos}] {video_id} handled {num_events} events. Saved array {frames.shape} in {t_file:.3f}s (read: {t_read:.2f}s, conv: {t_conv:.2f}s, save: {t_save:.2f}s)"
            return (True, msg)

        except Exception as e:
            return (False, f"[{i}/{total_videos}] Conversion crashed for {video_id}: {e}")

    # Use ThreadPoolExecutor or ProcessPoolExecutor depending on device
    # For GPU operations in PyTorch, threads are deeply bound, but ThreadPoolExecutor keeps GPU memory in a single process to avoid PyTorch multiprocessing CUDA sharing errors
    # For CPU, multiprocessing would be better, but given it runs fast on GPU, we use simple threads for I/O bound masking.
    workers = 4 # Default standard concurrent workers
    print(f"Starting parallel batch conversion with {workers} concurrent workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Map tasks
        results = executor.map(process_row, enumerate(rows, start=1))

        # Process results as they finish
        for is_success, msg in results:
            print(msg)
            if is_success:
                success += 1
            else:
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
