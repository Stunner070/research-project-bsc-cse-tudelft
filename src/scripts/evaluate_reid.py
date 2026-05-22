import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import json

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.datasets.event_video_dataset import EventVideoDataset
from src.models.model_factory import build_reid_model
from src.utils.metrics import evaluate_query_gallery
from src.scripts.train_compare_representations import get_device, FaceNetTransform, extract_embeddings

def main():
    parser = argparse.ArgumentParser(description="Evaluate face-ReID model with Query/Gallery protocol and ASR support.")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="Which split to evaluate.")
    parser.add_argument("--mode", type=str, choices=["clean_reid", "attack_reid"], default="clean_reid", help="Evaluation mode.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights.")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Model backbone used.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use.")
    parser.add_argument("--frames_root_clean", type=str, default=str(config.FRAMES_ROOT), help="Path to clean frames.")
    parser.add_argument("--frames_root_attack", type=str, default=None, help="Path to attacked frames.")
    # Assuming training used the enriched manifest to get num_classes, we might need a mapping or we can just ignore it for evaluation if the backbone does not need final layer for embeddings. Wait, the model creation needs `num_classes`. Let's read from the train split.
    parser.add_argument("--train_csv", type=str, default=str(config.SPLITS_DIR / "train.csv"))
    parser.add_argument("--eval_csv", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None, help="Optional CSV to save per-query ASR bookkeeping.")

    args = parser.parse_args()
    device = get_device(args.device)

    eval_csv = args.eval_csv if args.eval_csv else str(config.SPLITS_DIR / f"{args.split}.csv")

    # Check if necessary frames exist
    if args.mode == "attack_reid" and not args.frames_root_attack:
        print("Error: --frames_root_attack must be provided for attack_reid mode")
        sys.exit(1)

    print(f"Loading data mappings from {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    id_col = next(c for c in ['identity', 'identity_id', 'ytb_id'] if c in train_df.columns)
    unique_ids = train_df[id_col].unique()
    id_to_int = {identity: i for i, identity in enumerate(unique_ids)}
    num_classes = len(unique_ids)

    transform = FaceNetTransform() if args.backbone == 'facenet' else None

    # Load Gallery (Always Clean)
    gallery_dataset = EventVideoDataset(
        manifest_csv=eval_csv, mode="event_frames", frames_root=args.frames_root_clean,
        transform=transform, id_to_int=id_to_int, role='gallery', is_attacked=False
    )
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load Clean Query
    query_clean_dataset = EventVideoDataset(
        manifest_csv=eval_csv, mode="event_frames", frames_root=args.frames_root_clean,
        transform=transform, id_to_int=id_to_int, role='query', is_attacked=False
    )
    query_clean_loader = DataLoader(query_clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Validation Checks
    assert len(query_clean_dataset) > 0, "No query samples found!"
    assert len(gallery_dataset) > 0, "No gallery samples found!"

    q_identities = set(query_clean_dataset.df[id_col].unique())
    g_identities = set(gallery_dataset.df[id_col].unique())
    assert q_identities.issubset(g_identities), "Not all query identities have a matching gallery sample!"
    assert len(set(query_clean_dataset.df['video_id']).intersection(set(gallery_dataset.df['video_id']))) == 0, "Query and Gallery sample sets overlap!"

    print(f"Initializing {args.backbone} model...")
    model = build_reid_model(args.backbone, num_classes).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    model.eval()

    print("Extracting Gallery features...")
    g_features, g_labels = extract_embeddings(model, gallery_loader, device)

    print("Extracting Clean Query features...")
    qc_features, qc_labels = extract_embeddings(model, query_clean_loader, device)

    print("Evaluating Clean ReID...")
    clean_rank1, clean_mAP, clean_top1_correct = evaluate_query_gallery(qc_features, qc_labels, g_features, g_labels)

    # Logging setup
    print("\n==============================================")
    print(f"Split:               {args.split}")
    print(f"Mode:                {args.mode}")
    print(f"Number of IDs:       {len(q_identities)}")
    print(f"Number of queries:   {len(query_clean_dataset)}")
    print(f"Number of gallery:   {len(gallery_dataset)}")
    print("----------------------------------------------")
    print(f"Clean Rank-1:        {clean_rank1:.4f}")
    print(f"Clean mAP:           {clean_mAP:.4f}")

    if args.mode == "attack_reid":
        # Load Attacked Query
        query_attack_dataset = EventVideoDataset(
            manifest_csv=eval_csv, mode="event_frames", frames_root=args.frames_root_clean,
            frames_root_attacked=args.frames_root_attack,
            transform=transform, id_to_int=id_to_int, role='query', is_attacked=True
        )
        query_attack_loader = DataLoader(query_attack_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        assert len(query_clean_dataset) == len(query_attack_dataset), "Clean and Attacked query sets must have the same number of samples"
        assert all(query_clean_dataset.df['video_id'].values == query_attack_dataset.df['video_id'].values), "Clean and Attacked query rows do not match"

        print("\nExtracting Attacked Query features...")
        qa_features, qa_labels = extract_embeddings(model, query_attack_loader, device)

        print("Evaluating Attacked ReID...")
        attack_rank1, attack_mAP, attack_top1_correct = evaluate_query_gallery(qa_features, qa_labels, g_features, g_labels)

        # Successful attack = clean Rank-1 WAS correct (optional condition, but standard ASR drops this requirement or includes it.
        # User defined: "the attacked/protected query FAILS to retrieve the correct identity at Rank-1 from the gallery".
        # So "attacked Rank-1 is wrong for that query).
        attack_success = ~attack_top1_correct
        asr = attack_success.mean()

        print(f"Attacked Rank-1:     {attack_rank1:.4f}")
        print(f"Attacked mAP:        {attack_mAP:.4f}")
        print(f"ASR:                 {asr:.4f}")

        if args.output_csv:
            out_df = pd.DataFrame({
                'video_id': query_attack_dataset.df['video_id'],
                'identity': query_attack_dataset.df[id_col],
                'clean_rank1_correct': clean_top1_correct,
                'attacked_rank1_correct': attack_top1_correct,
                'attack_success': attack_success
            })
            out_df.to_csv(args.output_csv, index=False)
            print(f"Saved bookkeeping to {args.output_csv}")

    print("==============================================\n")


if __name__ == "__main__":
    main()

