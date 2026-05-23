from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent  # Base directory of the repository

# ─── Shared Paths (used by both pipelines) ───────────────────────────────────
V2E_ROOT = Path("/scratch/sofyanali/voxceleb/dev/baseline_346")  # Root directory with per-video v2e outputs (events.h5 + dvs.avi)
# V2E_ROOT = Path("C:/Users/sofya/Desktop/event_videos/346x260")
WORK_DIR = Path("/scratch/sofyanali/voxceleb/dev/output_baseline")  # Root directory for all derived outputs
# WORK_DIR = Path("C:/Users/sofya/Desktop/rp code/output")
MANIFEST_DIR = WORK_DIR / "manifests"  # Directory for storing generated dataset manifests
SPLITS_DIR = WORK_DIR / "splits"  # Directory for canonical train/val/test split CSVs
FRAMES_ROOT = WORK_DIR / "frames"  # Root directory for converted event-frame .npy arrays
CELEBVHQ_INFO_JSON = PROJECT_ROOT / "celebvhq_info.json"  # Path to external metadata mapping files
MODELS_EVENT_FRAMES = WORK_DIR / "models" / "event_frames"  # Directory to save event-frame model checkpoints
MODELS_DVS_AVI = WORK_DIR / "models" / "dvs_avi"  # Directory to save DVS AVI model checkpoints

# ─── Shared Sensor Settings (used by both pipelines) ─────────────────────────
EVENT_HEIGHT = 260  # Default pixel height for the DVS sensor representation
EVENT_WIDTH = 346  # Default pixel width for the DVS sensor representation
DEFAULT_DT = 5000.0  # Default microsecond integration time for event binning

# ─── Training Pipeline Only ──────────────────────────────────────────────────
DEFAULT_EPOCHS = 30  # Default number of training epochs
DEFAULT_BATCH_SIZE = 16  # Default batch size for data loaders
DEFAULT_NUM_WORKERS = 4  # Default number of multiprocessing workers for Dataloaders
# MODIFIED_QUERY_DIR = Path("C:/Users/sofya/Desktop/event_videos/346x260")
MODIFIED_QUERY_DIR = Path("/scratch/sofyanali/voxceleb/dev/baseline_346")  # Path to alternate queries for post-training ASR evaluation

# ─── Retrieval Evaluation Pipeline Only (no-training) ────────────────────────
RETRIEVAL_MANIFEST_A     = MANIFEST_DIR / "manifest_enriched.csv"  # Manifest path for baseline dataset
RETRIEVAL_FRAMES_ROOT_A  = FRAMES_ROOT  # Root frames path for baseline dataset
RETRIEVAL_LABEL_A        = "Baseline"  # Label name for baseline reporting

RETRIEVAL_MANIFEST_B     = MANIFEST_DIR / "manifest_enriched.csv"  # Manifest path for adjusted dataset
RETRIEVAL_FRAMES_ROOT_B  = FRAMES_ROOT  # Root frames path for adjusted dataset
RETRIEVAL_LABEL_B        = "Adjusted"  # Label name for adjusted reporting

RETRIEVAL_WEIGHTS_PATH   = PROJECT_ROOT / "models_weights" / "20180402-114759-vggface2.pt"  # Pretrained FaceNet backbone state dict
RETRIEVAL_OUTPUT_DIR     = WORK_DIR / "retrieval_results"  # Folder to write the comparison JSON
RETRIEVAL_BATCH_SIZE     = 32  # Inference batch size for retrieval dataloaders
