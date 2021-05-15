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
cd <PROJECT_DIRECTORY>/ssl

# SSL exps
python train_ssl.py --transform crop1 --exp_dir no_pretrain_crop1
python train_ssl.py --transform crop1 --pretrain --exp_dir pretrain_crop1
python train_ssl.py --transform crop2 --exp_dir no_pretrain_crop2
python train_ssl.py --transform crop2 --pretrain --exp_dir pretrain_crop2