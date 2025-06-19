#!/usr/bin/env python3
"""
KeyPoint-MoSeq Behavioral Analysis Pipeline

This script demonstrates the complete KeyPoint-MoSeq (KPMS) pipeline for discovering behavioral syllables from pose estimation data.

Pipeline steps:
1. Loads and formats pose data from CSV files
2. Performs PCA for dimensionality reduction
3. Fits AR-HMM model to discover behavioral syllables
4. Generates visualizations including trajectory plots and behavioral movies

Usage:
    python pipeline.py
"""
import os
import sys
import pathlib
import logging

# === Environment Configuration ===
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Optional: enable autoreload when running in IPython/Notebook
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip:
        ip.run_line_magic('load_ext', 'autoreload')
        ip.run_line_magic('autoreload', '2')
except ImportError:
    pass

import jax
print(f"Using device: {jax.devices()[0].platform}")

# === Project Path Setup ===
# Add project root (one level up) to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# === Imports ===
from src.utils import set_up_logging, print_gpu_usage, validate_data_quality
from src.methods import (
    load_and_format_data,
    perform_pca,
    fit_and_save_model,
    generate_plots_and_movies
)
from src.preprocessing import h5_to_csv_poses
import keypoint_moseq as kpms
from jax_moseq.utils import set_mixed_map_iters

# === Configuration Parameters ===
G_MIXED_MAP_ITERS = 8        # Reduce if running out of GPU memory
G_ARHMM_ITERS = 400          # AR-HMM iterations
G_FULL_MODEL_ITERS = 400     # Full model iterations
G_KAPPA = 0.1                # Stickiness parameter

# Bodyparts and skeleton definitions
G_BODYPARTS = [
    "NOSE_INDEX",
    "LEFT_EAR_INDEX",
    "RIGHT_EAR_INDEX",
    "BASE_NECK_INDEX",
    "LEFT_FRONT_PAW_INDEX",
    "RIGHT_FRONT_PAW_INDEX",
    "CENTER_SPINE_INDEX",
    "LEFT_REAR_PAW_INDEX",
    "RIGHT_REAR_PAW_INDEX",
    "BASE_TAIL_INDEX",
    "MID_TAIL_INDEX",
    "TIP_TAIL_INDEX",
]

G_SKELETON = [
    ["TIP_TAIL_INDEX", "MID_TAIL_INDEX"],
    ["MID_TAIL_INDEX", "BASE_TAIL_INDEX"],
    ["BASE_TAIL_INDEX", "RIGHT_REAR_PAW_INDEX"],
    ["BASE_TAIL_INDEX", "LEFT_REAR_PAW_INDEX"],
    ["BASE_TAIL_INDEX", "CENTER_SPINE_INDEX"],
    ["CENTER_SPINE_INDEX", "LEFT_FRONT_PAW_INDEX"],
    ["CENTER_SPINE_INDEX", "RIGHT_FRONT_PAW_INDEX"],
    ["CENTER_SPINE_INDEX", "BASE_NECK_INDEX"],
    ["BASE_NECK_INDEX", "NOSE_INDEX"],
]

# === Project Paths (update paths for your data) ===
G_BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
G_PROJ_NAME = "2025-06-12_kpms-training"
G_PROJ_PATH = os.path.join(G_BASE_PATH, "results")
G_VIDEO_DIR = os.path.join(G_BASE_PATH, "data", "videos")
G_POSE_DIR = os.path.join(G_BASE_PATH, "data", "poses")

# Optional CSV output directory for H5-to-CSV conversion
G_POSE_CSV_DIR = os.path.join(G_BASE_PATH, "data", "poses_csv")

# === Helper Functions ===
def initialize_project():
    """Initialize project settings and configurations for research analysis."""
    project_path = pathlib.Path(G_PROJ_PATH)

    # Create project directory if it doesn't exist
    project_path.mkdir(parents=True, exist_ok=True)

    # Set up KeyPoint-MoSeq project structure
    kpms.setup_project(
        project_path,
        video_dir=G_VIDEO_DIR,
        bodyparts=G_BODYPARTS,
        skeleton=G_SKELETON
    )

    # Configure logging
    log_dir = project_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    set_up_logging(log_dir)

    # Update project configuration
    kpms.update_config(
        project_path,
        anterior_bodyparts=["BASE_NECK_INDEX"],
        posterior_bodyparts=["BASE_TAIL_INDEX"],
        use_bodyparts=G_BODYPARTS
    )

    logging.info(f"Project initialized: {project_path}")
    logging.info(f"GPU memory config: {G_MIXED_MAP_ITERS} mixed map iters")
    return project_path

# === Main Pipeline ===
def main():
    # Set mixed map iterations for GPU optimization
    set_mixed_map_iters(G_MIXED_MAP_ITERS)
    print(f"Set mixed map iterations to {G_MIXED_MAP_ITERS}")

    # Initialize project (run once)
    project_path = initialize_project()

    # Optionally convert H5 pose files to CSV
    if not os.path.isdir(G_POSE_CSV_DIR):
        os.makedirs(G_POSE_CSV_DIR)

    if os.listdir(G_POSE_CSV_DIR):
        print(f"CSV files already exist in {G_POSE_CSV_DIR}, skipping conversion.")
    else:
        print(f"Converting H5 files in {G_POSE_DIR} to CSV format...")
        try:
            converted = h5_to_csv_poses(G_POSE_DIR, G_POSE_CSV_DIR)
            print(f"Successfully converted {len(converted)} files.")
        except Exception as e:
            logging.error(f"H5 to CSV conversion failed: {e}")
            print(f"Error during conversion: {e}")

    # Run full KPMS pipeline
    print("Starting KeyPoint-MoSeq analysis pipeline...")

    # Monitor initial GPU usage
    if jax.devices()[0].platform != 'cpu':
        print("\n=== Initial GPU Usage ===")
        print_gpu_usage()

    # Step 1: Load and format data
    print("\n=== Loading and Formatting Data ===")
    data, metadata, coords = load_and_format_data(G_POSE_CSV_DIR, project_path)

    # Optional: Data quality report
    # qrep = validate_data_quality(coords, metadata.get('confidences', {}))
    # for fname, metrics in qrep.items():
    #     print(f"{fname}: {metrics['total_frames']} frames, mean confidence: {metrics['mean_confidence']:.3f}")

    # Step 2: PCA
    print("\n=== Performing PCA Analysis ===")
    config_fn = lambda: kpms.load_config(project_path)
    pca, n_comp = perform_pca(data, config_fn, project_path)
    print(f"Components explaining >90% variance: {n_comp}")
    kpms.update_config(project_path, latent_dim=n_comp)

    if jax.devices()[0].platform != 'cpu':
        print("\n=== GPU Usage After PCA ===")
        print_gpu_usage()

    # Step 3: Fit AR-HMM model
    print(f"\n=== Fitting AR-HMM Model (kappa={G_KAPPA}) ===")
    model, model_name, results = fit_and_save_model(
        data, metadata, pca, config_fn, project_path,
        kappa=G_KAPPA,
        arhmm_iters=G_ARHMM_ITERS,
        full_model_iters=G_FULL_MODEL_ITERS
    )

    # Step 4: Generate visualizations and movies
    print("\n=== Generating Plots and Movies ===")
    generate_plots_and_movies(model_name, results, coords, project_path, config_fn)

    print("\n=== Analysis Complete! ===")
    print(f"Results saved to: {project_path}")
    print(f"Model name: {model_name}")

if __name__ == '__main__':
    main()
