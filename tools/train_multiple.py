import subprocess
import time
import os

# Define the parameters for multiple runs
num_bins_values = 15
x_shift_values = [0.7]
multiplier_values = [15]
eval_after_epoch = False
kd_loss_func = "entropyRelN" # "basic", "entropy", "entropyRelN"
top_n = 5000
top_n_relative = 0.75
gt_loss_weight = 1
kd_loss_weight = 1
epochs = 15
verbose = False

cfg_file = '/home/niko/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_second_S_w_teacher_multihead.yaml'
pretrained_model = None # pretrained student model
pretrained_model_teacher = '/home/niko/Documents/sicherung_trainings/second_2_240315/checkpoint_epoch_15.pth'
layer0_name_teacher = "backbone_3d.conv_out.0"
layer0_name_student = "backbone_3d.feat_adapt_single.0"
# Optional
layer1_name_teacher = None
layer1_name_student = None
layer2_name_teacher = None
layer2_name_student = None

current_date = time.strftime("%y%m%d")
# Define output_dir, if it exists, add a number to the name
i = 0
while os.path.exists(f"/home/niko/Documents/sicherung_trainings/multi_run_{current_date}_{i}"):
    i += 1
output_dir = f"/home/niko/Documents/sicherung_trainings/multi_run_{current_date}_{i}"
os.makedirs(f"{output_dir}/src", exist_ok=False)

# Save important .py files
subprocess.run(["cp", "train_kd.py", f"{output_dir}/src"])
subprocess.run(["cp", "./visual_utils/vis_feature_maps.py", f"{output_dir}/src"])
subprocess.run(["cp", "../pcdet/models/backbones_3d/spconv_backbone.py", f"{output_dir}/src"])

# Loop over the parameters
for k in range(1):
    # add a subdirectory for each run that is named after the parameter
    output_dir_num_bins = f"{output_dir}/kd_loss_{kd_loss_func}"
    os.makedirs(output_dir_num_bins, exist_ok=False)
    # create a file where all current training parameters are saved
    with open(f"{output_dir_num_bins}/training_parameters.txt", "w") as file:
        file.write(f"current_date: {current_date}\n")
        file.write(f"cfg_file: {cfg_file}\n")
        file.write(f"pretrained_model: {pretrained_model}\n")
        file.write(f"pretrained_model_teacher: {pretrained_model_teacher}\n")
        file.write(f"epochs: {epochs}\n")
        file.write(f"num_bins: {num_bins_values}\n")
        file.write(f"x_shift: {x_shift_values[0]}\n")
        file.write(f"multiplier: {multiplier_values[0]}\n")
        file.write(f"eval_after_epoch: {eval_after_epoch}\n")
        file.write(f"kd_loss_func: {kd_loss_func}\n")
        file.write(f"top_n: {top_n}\n")
        file.write(f"top_n_relative: {top_n_relative}\n")
        file.write(f"gt_loss_weight: {gt_loss_weight}\n")
        file.write(f"kd_loss_weight: {kd_loss_weight}\n")
        file.write(f"layer0_name_teacher: {layer0_name_teacher}\n")
        file.write(f"layer0_name_student: {layer0_name_student}\n")
        file.write(f"layer1_name_teacher: {layer1_name_teacher}\n")
        file.write(f"layer1_name_student: {layer1_name_student}\n")
        file.write(f"layer2_name_teacher: {layer2_name_teacher}\n")
        file.write(f"layer2_name_student: {layer2_name_student}\n")

    # Define the command as a list
    cmd = ["python", "train_kd.py",
           "--cfg_file", cfg_file,
           "--pretrained_model_teacher", pretrained_model_teacher,
           "--epochs", str(epochs),
           "--num_bins", str(num_bins_values),
           "--output_dir", output_dir_num_bins,
           "--kd_loss_func", kd_loss_func,
           "--top_n", str(top_n),
           "--top_n_relative", str(top_n_relative),
           "--gt_loss_weight", str(gt_loss_weight),
           "--kd_loss_weight", str(kd_loss_weight),
           "--x_shift", str(x_shift_values[0]),
           "--multiplier", str(multiplier_values[0]),
           "--layer0_name_teacher", layer0_name_teacher,
           "--layer0_name_student", layer0_name_student]

    # None values are converted to strings "None", which creates errors
    if layer1_name_teacher is not None:
        cmd.extend(["--layer1_name_teacher", layer1_name_teacher])
    if layer1_name_student is not None:
        cmd.extend(["--layer1_name_student", layer1_name_student])
    if layer2_name_teacher is not None:
        cmd.extend(["--layer2_name_teacher", layer2_name_teacher])
    if layer2_name_student is not None:
        cmd.extend(["--layer2_name_student", layer2_name_student])
    if pretrained_model is not None:
        cmd.extend(["--pretrained_model", pretrained_model])
    if eval_after_epoch:
        cmd.append("--eval_after_epoch")
    if verbose:
        cmd.append("--v")

    print("Running with: ", cmd)
    # Run the command
    subprocess.run(cmd)