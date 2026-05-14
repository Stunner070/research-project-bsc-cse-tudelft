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
    id_column: str = "ytb_id",
    min_clips_per_identity: int = 2,
    seed: int = 42
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

    # Analyze Clips per Identity
    counts = df['identity'].value_counts()
    identities_gt_1 = (counts >= 1).sum()
    identities_gt_2 = (counts >= 2).sum()
    identities_gt_3 = (counts >= 3).sum()
    identities_gt_4 = (counts >= 4).sum()

    print("\n--- Identity/Clip Distribution ---")
    print(f"Total initial samples: {len(df)}")
    print(f"Total initial identities: {len(counts)}")
    print(f"Identities with >=1 clips: {identities_gt_1}")
    print(f"Identities with >=2 clips: {identities_gt_2}")
    print(f"Identities with >=3 clips: {identities_gt_3}")
    print(f"Identities with >=4 clips: {identities_gt_4}")

    # Filter out identities with < min_clips_per_identity
    valid_identities = counts[counts >= min_clips_per_identity].index.tolist()
    initial_sample_count = len(df)
    initial_id_count = len(counts)

    df = df[df['identity'].isin(valid_identities)].copy()

    retained_sample_count = len(df)
    retained_id_count = len(valid_identities)

    print(f"\nFiltering identities with < {min_clips_per_identity} clips...")
    print(f"Removed {initial_id_count - retained_id_count} identities.")
    print(f"Removed {initial_sample_count - retained_sample_count} samples.")
    print(f"Retained {retained_sample_count} samples across {retained_id_count} identities.\n")

    unique_ids = list(df['identity'].unique())
    np.random.seed(seed)
    np.random.shuffle(unique_ids)

    n_ids = len(unique_ids)
    n_train = int(0.7 * n_ids)
    n_val = int(0.15 * n_ids)
    n_test = n_ids - n_train - n_val

    # Safety Fallbacks for very small datasets
    if n_val == 0 and n_ids >= 2:
        n_val = 1
        n_train -= 1
    if n_test == 0 and n_ids >= 3:
        n_test = 1
        n_train -= 1

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train+n_val]
    test_ids = unique_ids[n_train+n_val:]

    # Assertions
    assert len(set(train_ids) & set(val_ids)) == 0, "Train and Val identity sets overlap!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train and Test identity sets overlap!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val and Test identity sets overlap!"

    train_df = df[df['identity'].isin(train_ids)]
    val_df = df[df['identity'].isin(val_ids)]
    test_df = df[df['identity'].isin(test_ids)]

    # Assign Query / Gallery roles
    def assign_roles(split_df):
        if split_df.empty: return split_df
        split_df = split_df.copy()
        split_df['role'] = 'gallery'
        
        # Pick 1 random query per identity
        np.random.seed(seed)
        query_indices = []
        for _, group in split_df.groupby('identity'):
            query_indices.append(np.random.choice(group.index))
            
        split_df.loc[query_indices, 'role'] = 'query'
        return split_df

    train_df = train_df.copy()
    train_df['role'] = 'train'
    val_df = assign_roles(val_df)
    test_df = assign_roles(test_df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Not all samples are allocated exactly once!"

    # Combine back to df for output_enriched
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    output_enriched.parent.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_enriched, index=False)

    if not train_df.empty: train_df.to_csv(splits_dir / "train.csv", index=False)
    if not val_df.empty: val_df.to_csv(splits_dir / "val.csv", index=False)
    if not test_df.empty: test_df.to_csv(splits_dir / "test.csv", index=False)

    print("--- Final Split Summary ---")
    print(f"Identity source used: {id_column}")
    print(f"Min clips per identity: {min_clips_per_identity}")
    print(f"Total retained samples: {len(df)}")
    print(f"Total retained identities: {len(unique_ids)}")
    print(f"Train: {len(train_df):>5} samples, {len(train_ids):>5} identities")
    print(f"Val:   {len(val_df):>5} samples, {len(val_ids):>5} identities")
    print(f"Test:  {len(test_df):>5} samples, {len(test_ids):>5} identities")
    print("Confirmed: Identity sets are strictly disjoint.\n")

    return int(len(df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_enriched", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--id_column", type=str, default="ytb_id")
    parser.add_argument("--min_clips_per_identity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(
        Path(args.manifest_csv),
        Path(args.json_path),
        Path(args.output_enriched),
        Path(args.splits_dir),
        args.id_column,
        args.min_clips_per_identity,
        args.seed
    )
