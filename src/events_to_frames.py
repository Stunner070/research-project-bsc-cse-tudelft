import argparse
import h5py
import numpy as np
import math
import torch
from pathlib import Path
from typing import Union, Optional

def load_events(h5_path: Union[str, Path]) -> np.ndarray:
    """
    Opens the HDF5 file, reads the 'events' dataset into a NumPy array of dtype float32,
    and returns an array of shape (N, 4) -> [t, x, y, p].
    """
    with h5py.File(str(h5_path), "r") as f:
        events = np.array(f["events"], dtype=np.float32)
    return events

def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def events_to_frames(events: np.ndarray,
                     H: int,
                     W: int,
                     dt: float,
                     t0: Optional[float] = None,
                     t1: Optional[float] = None,
                     device_arg: str = "auto") -> np.ndarray:
    """
    Converts events into a sequence of frames of shape (T, H, W).
    Uses PyTorch for fully vectorized GPU/CPU accumulation.
    """
    if len(events) == 0:
        return np.zeros((0, H, W), dtype=np.float32)

    if t0 is None:
        t0 = float(events[:, 0].min())
    if t1 is None:
        t1 = float(events[:, 0].max())

    T = math.ceil((t1 - t0) / dt)
    if T <= 0:
        return np.zeros((0, H, W), dtype=np.float32)

    device = get_device(device_arg)
    
    # Initialize the output sequence directly on the appropriate device
    frames = torch.zeros((T, H, W), dtype=torch.float32, device=device)

    # Convert the entire events array to a tensor on the device for fast slicing
    events_tensor = torch.from_numpy(events).to(device)

    # Completely vectorized binning without Python loops
    t = events_tensor[:, 0]
    x = events_tensor[:, 1].long()
    y = events_tensor[:, 2].long()
    p = events_tensor[:, 3]

    # Calculate time bins directly
    k = torch.floor((t - t0) / dt).long()

    # Filter out spatial bounds and invalid time bins
    valid_mask = (k >= 0) & (k < T) & (x >= 0) & (x < W) & (y >= 0) & (y < H)

    k_valid = k[valid_mask]
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    p_valid = p[valid_mask]

    if len(k_valid) > 0:
        # Signed contribution
        p_signed = torch.where(p_valid == 1.0, 1.0, -1.0)
        # Accumulate using index_put_ onto the full 3D spatiotemporal tensor in one shot
        frames.index_put_((k_valid, y_valid, x_valid), p_signed, accumulate=True)

    # Return the accumulated frames sequence back to CPU as a NumPy array
    return frames.cpu().numpy()

def save_frames_npy(frames: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Saves the numpy array of frames as a .npy file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, frames)

def main():
    parser = argparse.ArgumentParser(description="Convert v2e raw events to a sequence of event-frames.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the input events.h5 file.")
    parser.add_argument("--output_npy", type=str, required=True, help="Path to the output .npy file.")
    parser.add_argument("--dt", type=float, required=True, help="Time window size for each frame (in the same units as t).")
    parser.add_argument("--height", type=int, default=768, help="Frame height (default: 768).")
    parser.add_argument("--width", type=int, default=1024, help="Frame width (default: 1024).")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to use for processing.")
    args = parser.parse_args()

    # Log device
    resolved_device = get_device(args.device)
    print(f"events_to_frames using device: {resolved_device} (Args: {args.device})")

    events = load_events(args.h5_path)
    frames = events_to_frames(events, H=args.height, W=args.width, dt=args.dt, device_arg=args.device)
    save_frames_npy(frames, args.output_npy)

    print(f"Successfully converted {len(events)} events into {frames.shape[0]} frames of shape ({args.height}, {args.width}).")
    print(f"Saved generated frames to {args.output_npy}")

if __name__ == "__main__":
    main()
