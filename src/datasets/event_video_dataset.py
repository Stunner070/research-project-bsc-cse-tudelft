import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


class EventVideoDataset(Dataset):
    def __init__(self, manifest_csv, mode="event_frames", frames_root=None, transform=None, max_frames=None, id_to_int=None, role=None, frames_root_attacked=None, is_attacked=False):
        self.mode = mode
        self.frames_root = Path(frames_root) if frames_root else None
        self.frames_root_attacked = Path(frames_root_attacked) if frames_root_attacked else None
        self.is_attacked = is_attacked
        self.transform = transform
        self.max_frames = max_frames

        self.df = pd.read_csv(manifest_csv)

        if role is not None:
            self.df = self.df[self.df['role'] == role].copy()

        # Build mapping from identity_id to integer label if not provided
        id_col = 'identity' if 'identity' in self.df.columns else ('identity_id' if 'identity_id' in self.df.columns else 'video_id')
        self.id_col = id_col

        # Drop missing files for event_frames mode
        if self.mode == "event_frames" and self.frames_root:
            exists = self.df['video_id'].apply(lambda vid: (self.frames_root / vid / "event_frames.npy").exists())
            if not exists.all():
                print(f"Warning: dropping {sum(~exists)} samples missing event_frames.npy")
                self.df = self.df[exists].copy()

        if len(self.df) == 0:
            raise ValueError(f"No valid samples remaining in {manifest_csv} for role {role}")

        unique_ids = sorted(self.df[id_col].unique())
        if id_to_int is None:
            self.id_to_int = {identity: i for i, identity in enumerate(unique_ids)}
        else:
            self.id_to_int = id_to_int

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row['video_id']
        # If testing, label might not be in id_to_int, just assign -1 in that case for evaluation
        label = self.id_to_int.get(row[self.id_col], -1)

        if self.mode == "event_frames":
            root = self.frames_root_attacked if self.is_attacked else self.frames_root
            npy_path = root / video_id / "event_frames.npy"

            with open(npy_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)

                T = shape[0]

                if T == 0:
                    # Fallback: use configured event resolution
                    frame = np.zeros((config.EVENT_HEIGHT, config.EVENT_WIDTH), dtype=np.float32)
                else:
                    # Deterministic middle-frame selection
                    t = T // 2

                    data_offset = f.tell()
                    frame_bytes = np.dtype(dtype).itemsize * shape[1] * shape[2]
                    f.seek(data_offset + t * frame_bytes)

                    frame = np.fromfile(f, dtype=dtype, count=shape[1] * shape[2])
                    if frame.size < shape[1] * shape[2]:
                        frame = np.zeros((shape[1], shape[2]), dtype=np.float32)
                    else:
                        frame = frame.reshape((shape[1], shape[2]))

            # Safe normalization to [0, 1]
            fmin = frame.min()
            fmax = frame.max()
            frame = (frame - fmin) / (fmax - fmin + 1e-8)

            frame_tensor = torch.from_numpy(frame.copy()).unsqueeze(0).float()

        elif self.mode == "dvs_avi":
            dvs_path = row['dvs_avi_path']
            # Open video with OpenCV
            cap = cv2.VideoCapture(str(dvs_path))
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
                frame_tensor = torch.zeros((1, config.EVENT_HEIGHT, config.EVENT_WIDTH), dtype=torch.float32)
            else:
                # Deterministic middle-frame selection
                t = len(all_frames) // 2

                frame = all_frames[t].astype(np.float32)
                # Normalize to [0, 1]
                fmin = frame.min()
                fmax = frame.max()
                frame = (frame - fmin) / (fmax - fmin + 1e-8)
                frame_tensor = torch.from_numpy(frame).unsqueeze(0).float()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.transform:
            frame_tensor = self.transform(frame_tensor)

        return frame_tensor, label, video_id
