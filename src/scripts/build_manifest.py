import argparse
import csv
from pathlib import Path

def build_manifest(root: Path, output_csv: Path) -> int:
    root_path = root.resolve()
    output_path = output_csv.resolve()

    manifest_entries = []

    # Recursively find all events.h5 files
    for events_path in root_path.rglob("events.h5"):
        parent_dir = events_path.parent
        video_id = parent_dir.name
        identity_id = video_id

        dvs_avi_path = parent_dir / "dvs.avi"
        dvs_avi_str = str(dvs_avi_path.resolve()) if dvs_avi_path.exists() else ""

        manifest_entries.append({
            "video_id": video_id,
            "identity_id": identity_id,
            "events_path": str(events_path.resolve()),
            "dvs_avi_path": dvs_avi_str
        })

    # Write the sorted manifest to the output CSV
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
    args = parser.parse_args()

    build_manifest(Path(args.root), Path(args.output_csv))

if __name__ == "__main__":
    main()
