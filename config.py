from pathlib import Path

# Base directories (LOCAL TEST SETTINGS)
PROJECT_ROOT = Path(__file__).resolve().parent

# Root directory with per-video v2e outputs (events.h5 + dvs.avi)
V2E_ROOT = Path("/scratch/sofyanali/celebvhq/first_500_conv/baseline_346") # TODO: set this once

# Root directory for all derived outputs
WORK_DIR = Path("/home/sofyanali/projects/rpbsc/output") # TODO: set this once

# Info json
CELEBVHQ_INFO_JSON = PROJECT_ROOT / "celebvhq_info.json"

# Derived paths
MANIFEST_DIR = WORK_DIR / "manifests"
SPLITS_DIR = WORK_DIR / "splits"
FRAMES_ROOT = WORK_DIR / "frames"
MODELS_EVENT_FRAMES = WORK_DIR / "models" / "event_frames"
MODELS_DVS_AVI = WORK_DIR / "models" / "dvs_avi"

# Representation / training defaults
EVENT_HEIGHT = 768
EVENT_WIDTH = 1024
DEFAULT_DT = 5000.0
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 0
