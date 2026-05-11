import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

def build_splits(
    manifest_csv: Path,
    json_path: Path,
    output_enriched: Path,
    splits_dir: Path,
    id_column: str = "ytb_id"
):
    print(f"Reading manifest from {manifest_csv}")
    df = pd.read_csv(manifest_csv)

    # We expect video_id to be something like --aqjaJyZLk_0
    print(f"Reading JSON from {json_path}")
    if not json_path.exists():
        print(f"Warning: JSON not found at {json_path}. Mocking ytb_id with video_id for now.")
        df['ytb_id'] = df['video_id']
        df['clip_id'] = df['video_id']
        df['bbox_top'] = 0
        df['bbox_bottom'] = config.EVENT_HEIGHT
        df['bbox_left'] = 0
        df['bbox_right'] = config.EVENT_WIDTH
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        clips = data.get("clips", {})

        info_list = []
        for vid in df['video_id']:
            clip_info = clips.get(vid)
            if clip_info:
                bbox = clip_info.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) < 4:
                    bbox = [0, 0, config.EVENT_WIDTH, config.EVENT_HEIGHT]

                info_list.append({
                    "video_id": vid,
                    "clip_id": vid,
                    "ytb_id": clip_info.get("ytb_id", vid),
                    "bbox_top": bbox[1],
                    "bbox_bottom": bbox[3],
                    "bbox_left": bbox[0],
                    "bbox_right": bbox[2],
                })
            else:
                # Fallback, extract ytb_id by removing the last underscore index if standard format
                ytb_id = vid.rsplit('_', 1)[0] if '_' in vid else vid
                info_list.append({
                    "video_id": vid,
                    "clip_id": vid,
                    "ytb_id": ytb_id,
                    "bbox_top": 0,
                    "bbox_bottom": config.EVENT_HEIGHT,
                    "bbox_left": 0,
                    "bbox_right": config.EVENT_WIDTH,
                })

        info_df = pd.DataFrame(info_list)
        df = df.merge(info_df, on="video_id", how="left")

    df['identity'] = df[id_column]

    unique_ids = df['identity'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)

    n_ids = len(unique_ids)
    n_train = max(1, int(0.7 * n_ids))
    n_val = max(0, int(0.15 * n_ids))

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train+n_val]
    test_ids = unique_ids[n_train+n_val:]

    if len(val_ids) == 0 and n_ids > 1:
        val_ids = test_ids
        test_ids = []

    train_df = df[df['identity'].isin(train_ids)]
    val_df = df[df['identity'].isin(val_ids)]
    test_df = df[df['identity'].isin(test_ids)]

    output_enriched.parent.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_enriched, index=False)

    if not train_df.empty: train_df.to_csv(splits_dir / "train.csv", index=False)
    if not val_df.empty: val_df.to_csv(splits_dir / "val.csv", index=False)
    if not test_df.empty: test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"Total samples: {len(df)}")
    print(f"Total identities: {len(unique_ids)}")
    print(f"Identity source used: {id_column}")
    print(f"Train: {len(train_df)} samples, {len(train_ids)} identities")
    print(f"Val: {len(val_df)} samples, {len(val_ids)} identities")
    print(f"Test: {len(test_df)} samples, {len(test_ids)} identities")

    return int(len(df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_enriched", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--id_column", type=str, default="ytb_id")
    args = parser.parse_args()

    build_splits(
        Path(args.manifest_csv),
        Path(args.json_path),
        Path(args.output_enriched),
        Path(args.splits_dir),
        args.id_column
    )
