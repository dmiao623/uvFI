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
#SBATCH --output=/projects/kumar-lab/miaod/projects/uvFI/experiments/2025-06-10_kpms-inference/logs/output-%j.out

#conda activate keypoint_moseq
python3 preprocess_gs.py
