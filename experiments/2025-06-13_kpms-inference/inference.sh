#!/bin/bash
#
#SBATCH --job-name=unsupervised_inference
#
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_inference
#SBATCH --partition=gpu_a100
#SBATCH --mem=126G
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --array=1-79

#conda init
#conda activate keypoint_moseq_gs
#module load singularity
#singularity exec /projects/kumar-lab/sabnig/Builds/PyBase2.sif \
python3 inference_gs.py $SLURM_ARRAY_TASK_ID
