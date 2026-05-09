import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class EventVideoDataset(Dataset):
    def __init__(self, manifest_csv, mode="event_frames", frames_root=None, transform=None, max_frames=None):
        self.mode = mode
        self.frames_root = Path(frames_root) if frames_root else None
        self.transform = transform
        self.max_frames = max_frames

        self.df = pd.read_csv(manifest_csv)

        # Build mapping from identity_id to integer label
        unique_ids = self.df['identity_id'].unique()
        self.id_to_int = {identity: i for i, identity in enumerate(unique_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']
        label = self.id_to_int[row['identity_id']]

        if self.mode == "event_frames":
            npy_path = self.frames_root / video_id / "event_frames.npy"
            frames = np.load(npy_path)
            T = frames.shape[0]

            if T == 0:
                frame_tensor = torch.zeros((1, 768, 1024), dtype=torch.float32)
            else:
                if self.max_frames is not None and T > self.max_frames:
                    # Randomly sample max_frames indices
                    indices = np.random.choice(T, self.max_frames, replace=False)
                    # For now, return a single randomly chosen frame from the sampled indices
                    t = np.random.choice(indices)
                else:
                    t = np.random.randint(0, T)

                frame = frames[t]
                # Convert to (1, H, W) tensor
                frame_tensor = torch.from_numpy(frame).unsqueeze(0).float()

        elif self.mode == "dvs_avi":
            dvs_path = row['dvs_avi_path']
            # Open video with OpenCV
            cap = cv2.VideoCapture(dvs_path)
            all_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_frames.append(gray)

            cap.release()

            if len(all_frames) == 0:
                frame_tensor = torch.zeros((1, 768, 1024), dtype=torch.float32)
            else:
                if self.max_frames is not None and len(all_frames) > self.max_frames:
                    indices = np.random.choice(len(all_frames), self.max_frames, replace=False)
                    t = np.random.choice(indices)
                else:
                    t = np.random.randint(0, len(all_frames))

                frame = all_frames[t]
                frame_tensor = torch.from_numpy(frame).unsqueeze(0).float()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.transform:
            frame_tensor = self.transform(frame_tensor)

        return frame_tensor, label

