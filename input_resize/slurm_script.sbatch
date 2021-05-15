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
# Navigate to project directory
cd <PROJECT_DIRECTORY>/input_resize

python train_resize.py --transform crop1 --exp_dir resize_crop1
python train_resize.py --transform crop2 --exp_dir resize_crop2
python train_resize.py --transform crop3 --exp_dir resize_crop3
python train_resize.py --transform crop4 --exp_dir resize_crop4