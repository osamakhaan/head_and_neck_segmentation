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
# navigate to the directory
cd <PROJECT_DIRECTORY>/augmentation_model

# Baseline
python train_augmentation.py --model unet --num_epochs 250 --exp_dir baseline

# Augmentation exps
python train_augmentation.py --model unet --num_epochs 250 --transform rand_affine --exp_dir aug_exp_rand_affine
python train_augmentation.py --model unet --num_epochs 250 --transform rand_elastic --exp_dir aug_exp_rand_elastic
python train_augmentation.py --model unet --num_epochs 250 --transform rand_spatial_crop --exp_dir aug_exp_rand_spatial_crop
python train_augmentation.py --model unet --num_epochs 250 --transform rand_zoom --exp_dir aug_exp_rand_zoom

# Delayed augmentation exps
python train_augmentation.py --model unet --num_epochs 250 --transform rand_affine --delay --exp_dir aug_delayed_rand_affine
python train_augmentation.py --model unet --num_epochs 250 --transform rand_elastic --delay --exp_dir aug_delayed_rand_elastic
python train_augmentation.py --model unet --num_epochs 250 --transform rand_spatial_crop --delay --exp_dir aug_delayed_rand_spatial_crop
python train_augmentation.py --model unet --num_epochs 250 --transform rand_zoom --delay --exp_dir aug_delayed_rand_zoom

# Model exps
python train_augmentation.py --model unet --num_epochs 250 --exp_dir model_exp_normal_unet
python train_augmentation.py --model deeper_unet --num_epochs 250 --exp_dir model_exp_deeper_unet
python train_augmentation.py --model segresnet --num_epochs 250 --exp_dir model_exp_segresnet