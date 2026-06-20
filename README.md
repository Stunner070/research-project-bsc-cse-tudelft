# TU Delft BSc CSE Thesis Project: Event Camera Retrieval & Privacy Evaluation Pipeline

Repository used for my thesis project at the TU Delft Bsc Computer Science and Engineering.

## Project Overview

This project implements a **Retrieval Evaluation Pipeline** designed to evaluate the privacy and identifiability of individuals in event camera data. It processes event streams (generated via `v2e` from standard videos, like VoxCeleb) and compares baseline event representations against adjusted/degraded representations (such as altered resolutions or modified sensor leak parameters).

The pipeline executes sequentially in three distinct phases:
1. **Phase 1: Intrinsic Structural Evaluation:** Evaluates basic quality using SSIM/PSNR on `RAW_NPY` pre-computed event frames.
2. **Phase 2: Reconstruction Attack Evaluation:** Dynamically reconstructs events to frames using `E2VID` and performs ReID (e.g., via FaceNet) to measure identity leakage (calculating Rank-1 accuracy, mAP, and Attack Success Rate / ASR).
3. **Phase 3: Aggregation:** Merges the structural metrics and query retrieval results into detailed CSV outputs and JSON summaries.

It supports:
- **Facial Recognition (ReID):** Evaluates identifiability using pretrained models like **FaceNet** or **InsightFace**.
- **Structural Metrics:** Computes baseline image similarity metrics like SSIM/PSNR.
- **Face Cropping & Temporal Sampling:** Automatically crops faces using annotations or detectors, and samples frames using multiple strategies (e.g., center frame or multi-frame averaging).

## Directory Structure
- `config.py`: Main configuration file for paths, datasets, model selection, and pipeline parameters.
- `e2vid_config.py`: Configuration for the `E2VID` video reconstruction integration.
- `run_retrieval_pipeline.py`: The main entry point to execute the retrieval evaluation.
- `src/`: Contains the core logic for the pipeline.
  - `src/retrieval_eval/`: Contains scripts for embedding generation (`embed.py`), running the evaluation (`run_retrieval_eval.py`), and calculating metrics (`retrieval_metrics.py`).
- `configs/`: Directory for any YAML based configuration files.
- `models_weights/`: Directory meant to store pretrained weights (e.g., FaceNet backbone `20180402-114759-vggface2.pt`).

## Setup Process

### 1. Prerequisites
Ensure you have Python installed (e.g., Python 3.8+). It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```
*(Note: If you plan to use InsightFace (`RETRIEVAL_MODEL_NAME = "insightface"`), uncomment `insightface` and `onnxruntime-gpu` in `requirements.txt` before installing.)*

### 3. Model Weights
The pipeline relies on pretrained model weights:
- For **FaceNet**, ensure `20180402-114759-vggface2.pt` is placed inside the `models_weights/` directory.
- For **E2VID** reconstruction (if used), ensure the E2VID repository and the `E2VID_lightweight.pth.tar` weights are accessible and properly referenced.

### 4. Configuration
Before running the pipeline, you **must** update the configuration files to point to your local directories or cluster paths:

**`config.py`**
- Update `V2E_ROOT` and `WORK_DIR` to point to your event data and derived output directories.
- Define your datasets (Dataset A for Baseline, Dataset B for Adjusted) by modifying `RETRIEVAL_V2E_ROOT_A`, `RETRIEVAL_WORK_DIR_A` and their `_B` counterparts.
- (Note: `RETRIEVAL_PIPELINE_MODE` is typically left as `"RAW_NPY"`, as the pipeline automatically overrides it to `"E2VID"` during Phase 2).
- Select your embedding backend via `RETRIEVAL_MODEL_NAME` (`"facenet"` or `"insightface"`).

**`e2vid_config.py`** *(If using E2VID mode)*
- Update `E2VID_REPO_PATH` and `E2VID_MODEL_PATH` to point to your local installation of the `rpg_e2vid` repository.

## Usage

Once your environment is set up and configuration files are pointing to your local data, you can run the evaluation pipeline by executing:

```bash
python run_retrieval_pipeline.py
```

The pipeline will process the datasets sequentially through Phase 1 (structural metrics on raw events) and Phase 2 (ReID on E2VID reconstructions), calculate the differences, and output the retrieval results (e.g., JSON comparison files and raw metric CSVs) to the directory specified in `config.py` (by default, `RETRIEVAL_OUTPUT_DIR`).
