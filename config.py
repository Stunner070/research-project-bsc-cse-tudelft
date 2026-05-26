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
# Dataset A (Baseline) — source and output roots
# RETRIEVAL_V2E_ROOT_A     = Path("C:/Users/sofya/Desktop/event_videos/voxceleb/baseline_346")    # V2E source root for baseline
RETRIEVAL_V2E_ROOT_A = Path("/scratch/sofyanali/voxceleb/dev/baseline_346")
# RETRIEVAL_WORK_DIR_A     = Path("C:/Users/sofya/Desktop/event_videos/voxceleb/output_baseline")  # Derived-output root for baseline
RETRIEVAL_WORK_DIR_A     = Path("/scratch/sofyanali/voxceleb/dev/output_baseline")  # Derived-output root for baseline

# Dataset A — derived paths (built from roots above)
RETRIEVAL_MANIFEST_DIR_A = RETRIEVAL_WORK_DIR_A / "manifests"
RETRIEVAL_FRAMES_ROOT_A  = RETRIEVAL_WORK_DIR_A / "frames"
RETRIEVAL_MANIFEST_A     = RETRIEVAL_MANIFEST_DIR_A / "manifest_enriched.csv"
RETRIEVAL_LABEL_A        = "Baseline"

# Dataset B (Adjusted) — source and output roots
# RETRIEVAL_V2E_ROOT_B     = Path("C:/Users/sofya/Desktop/event_videos/voxceleb/leak_5")     # V2E source root for adjusted
RETRIEVAL_V2E_ROOT_B = Path("/scratch/sofyanali/voxceleb/dev/resolution_640")
# RETRIEVAL_WORK_DIR_B     = Path("C:/Users/sofya/Desktop/event_videos/voxceleb/output_leak_5")  # Derived-output root for adjusted
RETRIEVAL_WORK_DIR_B = Path("/scratch/sofyanali/voxceleb/dev/output_resolution_640")  # Derived-output root for adjusted

# Dataset B — derived paths (built from roots above)
RETRIEVAL_MANIFEST_DIR_B = RETRIEVAL_WORK_DIR_B / "manifests"
RETRIEVAL_FRAMES_ROOT_B  = RETRIEVAL_WORK_DIR_B / "frames"
RETRIEVAL_MANIFEST_B     = RETRIEVAL_MANIFEST_DIR_B / "manifest_enriched.csv"
RETRIEVAL_LABEL_B        = "resolution_640"

# Shared retrieval settings
RETRIEVAL_PIPELINE_MODE  = "RAW_NPY"    # "RAW_NPY" (pre-computed .npy) | "E2VID" (dynamic reconstruct)
RETRIEVAL_MAX_CLIPS      = 300          # Maximum number of clips to process from v2e roots (set to None for all)
RETRIEVAL_WEIGHTS_PATH   = PROJECT_ROOT / "models_weights" / "20180402-114759-vggface2.pt"  # Pretrained FaceNet backbone state dict
RETRIEVAL_OUTPUT_DIR     = PROJECT_ROOT / "retrieval_results_resolution_640_300clips"  # Folder to write the comparison JSON (independent of either work dir)
RETRIEVAL_BATCH_SIZE     = 32  # Inference batch size for retrieval dataloaders

# ─── Retrieval Model & Feature Settings ──────────────────────────────────────
RETRIEVAL_USE_FACE_MODELS     = True            # True = FaceNet/InsightFace ReID, False = SSIM/PSNR Structural Metrics
RETRIEVAL_MODEL_NAME          = "facenet"        # Embedding backend: "facenet" | "insightface"
                                                 # "insightface" uses a frozen pretrained recognizer (no training)

# ─── Face Cropping Settings ──────────────────────────────────────────────────
RETRIEVAL_USE_FACE_CROP       = True            # True = crop face region before embedding extraction
RETRIEVAL_FACE_CROP_SOURCE    = "annotation"     # Crop source: "annotation" (manifest bbox) | "insightface" (detector)
RETRIEVAL_FACE_CROP_MARGIN    = 0.15             # Fractional margin to expand around the detected/annotated bbox
RETRIEVAL_MIN_FACE_SIZE       = 20               # Reject face crops smaller than this in pixels (width or height)

# Annotation Crop Settings (when RETRIEVAL_FACE_CROP_SOURCE == "annotation")
RETRIEVAL_ANNOTATION_H5       = Path("/scratch/sofyanali/voxceleb/dev/txt.h5")  # Path to generated annotation HDF5
RETRIEVAL_ANNOTATION_FALLBACK = "median"         # Missing frame fallback: "nearest", "median", "none"

# ─── Temporal Sampling Settings ──────────────────────────────────────────────
RETRIEVAL_TEMPORAL_MODE       = "multi_average"         # Frame selection: "center" (single T//2) | "multi_average"
RETRIEVAL_NUM_SAMPLE_FRAMES   = 5               # Number of frames to sample when mode is "multi_average"
RETRIEVAL_FRAME_SAMPLE_STRATEGY = "uniform"      # Sampling strategy: "uniform" (evenly spaced) | "center_window"

# ─── InsightFace-specific Settings (frozen inference only) ───────────────────
RETRIEVAL_INSIGHTFACE_MODEL    = "buffalo_l"              # InsightFace model pack name
RETRIEVAL_INSIGHTFACE_PROVIDER = "CUDAExecutionProvider"  # ONNX Runtime execution provider
RETRIEVAL_INSIGHTFACE_DET_SIZE = (320, 320)               # Detector input resolution

# ─── Debug / Visualization ───────────────────────────────────────────────────
RETRIEVAL_SAVE_DEBUG_CROPS     = False                          # Save example crops for visual inspection
RETRIEVAL_DEBUG_CROP_DIR       = PROJECT_ROOT / "retrieval_debug_crops"  # Output directory for debug crops
RETRIEVAL_DEBUG_CROP_MAX       = 50                             # Max number of debug crops saved per run

# ─── Evaluation Run ──────────────────────────────────────────────────────────
RETRIEVAL_RUN_NAME             = "resolution_640_300clips"                       # Name used to export the CSV file with raw metrics
