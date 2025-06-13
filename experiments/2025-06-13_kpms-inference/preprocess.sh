#!/bin/bash
#
#SBATCH --job-name=unsupervised_preprocessing
#
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_inference
#SBATCH --partition=gpu_a100_mig
#SBATCH --mem=126G
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

#conda activate keypoint_moseq
python3 preprocess_gs.py
