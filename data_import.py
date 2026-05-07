# import yaml
#
# with open("configs/default.yaml", "r") as f:
#     cfg = yaml.safe_load(f)
#
# DATA_ROOT = cfg["data_root"]
# print("Using data from:", DATA_ROOT)

import os
from pathlib import Path

DATA_ROOT = Path("/scratch/sofyanali/celebvhq/videos")