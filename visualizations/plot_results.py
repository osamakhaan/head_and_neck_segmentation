import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import os

save_dir = "plots/mean_dc"

exp_type = "aug_exp_"
exp_names = ["no_aug", "rand_affine", "rand_elastic", "rand_spatial_crop", "rand_zoom"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"]
    y_axis = f["Value"]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Augmentations")
plt.legend()
plt.savefig(os.path.join(save_dir, "augmentations.png"))
plt.close()

exp_type = "aug_delayed_exp_"
exp_names = ["no_aug", "rand_affine", "rand_elastic", "rand_spatial_crop", "rand_zoom"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"][:15]
    y_axis = f["Value"][:15]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Augmentations (Delayed)")
plt.legend()
plt.savefig(os.path.join(save_dir, "augmentations_delayed.png"))
plt.close()


exp_type = "distributed_exp_"
exp_names = ["1_gpu", "2_gpu", "4_gpu"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"]
    y_axis = f["Value"]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Data Parallelism")
plt.legend()
plt.savefig(os.path.join(save_dir, "data_parallelism.png"))
plt.close()

exp_type = "distributed_exp_"
exp_names = ["1_gpu", "2_gpu", "4_gpu"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    if exp=="1_gpu":
        epochs=4
    elif exp=="2_gpu":
        epochs=7
    elif exp=="4_gpu":
        epochs=13
    x_axis = f["Step"][:epochs]
    y_axis = f["Value"][:epochs]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Data Parallelism")
plt.legend()
plt.savefig(os.path.join(save_dir, "data_parallelism_same_steps.png"))
plt.close()


exp_type = "effective_bsz_exp_"
exp_names = ["2", "4", "8", "16"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"]
    y_axis = f["Value"]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Effective Batch Size (LR Scaling)")
plt.legend()
plt.savefig(os.path.join(save_dir, "effective_bsz_lr_scaling.png"))
plt.close()

exp_type = "effective_bsz_no_lr_scaling_v2_exp_"
exp_names = ["2", "4", "8", "16"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"]
    y_axis = f["Value"]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Effective Batch Size (No LR Scaling)")
plt.legend()
plt.savefig(os.path.join(save_dir, "effective_bsz_no_lr_scaling.png"))
plt.close()


exp_type = "effective_bsz_no_lr_scaling_v2_exp_"
exp_names = ["2", "4", "8", "16"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"][:15]
    y_axis = f["Value"][:15]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Effective Batch Size (No LR Scaling)")
plt.legend()
plt.savefig(os.path.join(save_dir, "effective_bsz_no_lr_scaling_150_epochs.png"))
plt.close()


exp_type = "model_exp_"
exp_names = ["normal_unet", "deeper_unet", "segresnet"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"]
    y_axis = f["Value"]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Segmentation Models")
plt.legend()
plt.savefig(os.path.join(save_dir, "models.png"))
plt.close()

exp_type = "ssl_exp_"
exp_names = ["no_pretrain", "pretrain"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"]
    y_axis = f["Value"]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Self Supervised Pretraining")
plt.legend()
plt.savefig(os.path.join(save_dir, "pretraining.png"))
plt.close()

exp_type = "resize_vs_performance_exp_"
exp_names = ["crop1", "crop2", "crop3", "crop4"]
plt.figure(dpi=1200)
for exp in exp_names:
    f = pd.read_csv(exp_type+exp+".csv")
    x_axis = f["Step"][:10]
    y_axis = f["Value"][:10]

    plt.plot(x_axis, y_axis, label=exp)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Resizing vs Performance Tradeoff")
plt.legend()
plt.savefig(os.path.join(save_dir, "resizing_performance_tradeoff.png"))
plt.close()

plt.figure(dpi=1200)
epoch_time_factor = [round(300/300,1), round(450/300,1), round(550/300,1), round(650/300,1), round(890/300,1)]
dcs = [0.684, 0.701, 0.685, 0.708, 0.710]
dc_gain = [(dcs[i]-dcs[0])*1e2 for i in range(len(dcs))]
print(dc_gain)

plt.plot(epoch_time_factor, dc_gain)
plt.xlabel("Epoch Time (x no_aug)")
plt.xticks(epoch_time_factor, [f'no_aug ({epoch_time_factor[0]})', f'zoom ({epoch_time_factor[1]})', f'spatial_crop ({epoch_time_factor[2]})', f'affine ({epoch_time_factor[3]})', f'elastic ({epoch_time_factor[4]})'], rotation=15)
plt.ylabel("Dice Score Gain (x 1e-2)")
plt.title("Performance Gain vs Runtime Tradeoff")
# plt.tick_params(axis='x', which='major', labelsize=8)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, "augmentations_time_vs_gain.png"))
plt.close()

