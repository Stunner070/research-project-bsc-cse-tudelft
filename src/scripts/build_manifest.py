import argparse
import csv
from pathlib import Path


def build_manifest(root: Path, output_csv: Path, frames_root: Path = None) -> int:
    """Build a CSV manifest by scanning v2e_root for all valid converted samples.

    Scans recursively for events.h5 files.  Infers identity_id from the
    video_id using the project convention: video_id.split('_', 1)[0].
    No hard-coded max on identities or clips.
    """
    root_path = root.resolve()
    output_path = output_csv.resolve()

    seen_video_ids = set()
    manifest_entries = []

    # --- primary source: events.h5 files under v2e_root ---
    for events_path in root_path.rglob("events.h5"):
        parent_dir = events_path.parent
        video_id = parent_dir.name

        if video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)

        # Derive person-level identity from the naming convention
        identity_id = video_id.split("_", 1)[0] if "_" in video_id else video_id

        dvs_avi_path = parent_dir / "dvs.avi"
        dvs_avi_str = str(dvs_avi_path.resolve()) if dvs_avi_path.exists() else ""

        manifest_entries.append({
            "video_id": video_id,
            "identity_id": identity_id,
            "events_path": str(events_path.resolve()),
            "dvs_avi_path": dvs_avi_str,
        })

    # --- secondary source: already-converted event_frames.npy in frames_root ---
    if frames_root is not None:
        frames_root_path = Path(frames_root).resolve()
        if frames_root_path.exists():
            for npy_path in frames_root_path.rglob("event_frames.npy"):
                video_id = npy_path.parent.name
                if video_id in seen_video_ids:
                    continue
                seen_video_ids.add(video_id)

                identity_id = video_id.split("_", 1)[0] if "_" in video_id else video_id

                manifest_entries.append({
                    "video_id": video_id,
                    "identity_id": identity_id,
                    "events_path": "",
                    "dvs_avi_path": "",
                })

    # Sort for reproducibility
    manifest_entries.sort(key=lambda e: e["video_id"])

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["video_id", "identity_id", "events_path", "dvs_avi_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for entry in manifest_entries:
            writer.writerow(entry)

    print(f"Manifest generation complete. Found {len(manifest_entries)} entries in {root_path}.")
    print(f"Manifest written to {output_path}.")
    return len(manifest_entries)


def main():
    parser = argparse.ArgumentParser(description="Build a CSV manifest of v2e event records.")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output manifest CSV.")
    parser.add_argument("--frames_root", type=str, default=None, help="Optional frames root for already-converted data.")
    args = parser.parse_args()

    build_manifest(Path(args.root), Path(args.output_csv),
                   frames_root=Path(args.frames_root) if args.frames_root else None)


if __name__ == "__main__":
    main()
