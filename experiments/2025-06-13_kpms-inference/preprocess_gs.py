import pathlib
import sys, os 
import logging

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.preprocessing import h5_to_csv_poses

G_BASE_PATH = "/projects/kumar-lab/miaod/projects/uvFI/experiments/2025-06-13_kpms-inference/" 
G_BASE_DATA = G_BASE_PATH + "data/videos"
G_POSE_DIR = G_BASE_DATA + "/poses"

G_POSE_CSV_DIR = G_BASE_DATA + "/poses_csv"

if os.path.exists(G_POSE_CSV_DIR):
    if os.listdir(G_POSE_CSV_DIR):
        logging.info(f"The directory {G_POSE_CSV_DIR} is not empty. Skipping this step.")
    else:
        logging.info(f"The directory {G_POSE_CSV_DIR} is empty. Running h5_to_csv_poses()...")
        h5_to_csv_poses(G_POSE_DIR, G_POSE_CSV_DIR)
else:
    logging.error(f"The directory {G_POSE_CSV_DIR} does not exist. Creating it...")
    os.makedirs(G_POSE_CSV_DIR)
    logging.info(f"Directory {G_POSE_CSV_DIR} created. Running h5_to_csv_poses()...")
    h5_to_csv_poses(G_POSE_DIR, G_POSE_CSV_DIR)


