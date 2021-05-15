#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
# Load modules
module purge
module load miniconda/3-4.8.2
nvidia-smi
cd
cd /scratch/mma525/dl-system/head_and_neck_segmentation/loss_functions
/scratch/mma525/miniconda3/envs/dlsys/bin/python train_lossFunctions.py --exp_dir tversky_loss --loss_fn tversky_loss