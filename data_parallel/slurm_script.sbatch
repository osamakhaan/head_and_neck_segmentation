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
cd <PROJECT_DIRECTORY>/data_parallel

# Data parallel exps
CUDA_VISIBLE_DEVICES=0 python train_data_parallel.py --num_epochs 150 --exp_dir distributed_exp_1_gpu
CUDA_VISIBLE_DEVICES=0,1 python train_data_parallel.py --num_epochs 300 --exp_dir distributed_exp_2_gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_data_parallel.py --num_epochs 600 --exp_dir distributed_exp_4_gpu
