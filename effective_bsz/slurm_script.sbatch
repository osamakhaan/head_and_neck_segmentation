#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=mok232
#SBATCH --mail-user=mok232@nyu.edu
#SBATCH --output=slurm_%j.out
OPT=$1
#command line argument
. ~/.bashrc
conda activate pytorch
# Navigate to the project directory
cd <PROJECT_DIRECTORY>/effective_bsz

# Effective bsz exps
python train_effective_bsz.py --effective_bsz 2 --num_epochs 150 --exp_dir effective_bsz_exp_2
python train_effective_bsz.py --effective_bsz 4 --num_epochs 300 --exp_dir effective_bsz_exp_4
python train_effective_bsz.py --effective_bsz 8 --num_epochs 600 --exp_dir effective_bsz_exp_8
python train_effective_bsz.py --effective_bsz 16 --num_epochs 1200 --exp_dir effective_bsz_exp_16