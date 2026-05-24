import sys
from pathlib import Path

# E2VID Repository Path
# E2VID_REPO_PATH = "C:/Users/sofya/Desktop/rpg_e2vid-master"
E2VID_REPO_PATH = "/scratch/sofyanali/e2vid"  # Delft Blue cluster placeholder

# Path to pretrained model weights
# E2VID_MODEL_PATH = "C:/Users/sofya/Desktop/rpg_e2vid-master/pretrained/E2VID_lightweight.pth.tar"
E2VID_MODEL_PATH = "/scratch/sofyanali/e2vid/pretrained/E2VID_lightweight.pth.tar"

# Reconstruction parameters
config_dict = {
    "unsharp_mask_amount": 1.0,
    "unsharp_mask_sigma": 1.0,
    "bilateral_filter_sigma": 0.0,
    "fixed_duration": False,
    "window_size": 10000,
    "num_events_per_pixel": 0.35,
    "auto_hdr": False,
    "Imin": 0.0,
    "Imax": 1.0,
    "store_to_ram": True
}
