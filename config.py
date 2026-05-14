from pathlib import Path

# Base directories (LOCAL TEST SETTINGS)
PROJECT_ROOT = Path(__file__).resolve().parent

# Root directory with per-video v2e outputs (events.h5 + dvs.avi)
V2E_ROOT = Path("/scratch/sofyanali/celebvhq/useful_conv/baseline_346/")
# V2E_ROOT = Path("C:/Users/sofya/Desktop/event_videos/346x260")

# Root directory for all derived outputs
WORK_DIR = Path("/scratch/sofyanali/celebvhq/output/")
# WORK_DIR = Path("C:/Users/sofya/Desktop/rp code/output")

# Info json
CELEBVHQ_INFO_JSON = PROJECT_ROOT / "celebvhq_info.json"

# Derived paths
MANIFEST_DIR = WORK_DIR / "manifests"
SPLITS_DIR = WORK_DIR / "splits"
FRAMES_ROOT = WORK_DIR / "frames"
MODELS_EVENT_FRAMES = WORK_DIR / "models" / "event_frames"
MODELS_DVS_AVI = WORK_DIR / "models" / "dvs_avi"

# Representation / training defaults
EVENT_HEIGHT = 260
EVENT_WIDTH = 346
DEFAULT_DT = 5000.0
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
