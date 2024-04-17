import subprocess
import time
import os

# Define the parameters for multiple runs
num_bins_values = [5, 15, 45]
x_shift_values = [0.7]
multiplier_values = [15]
eval_after_epoch = False

current_date = time.strftime("%y%m%d")
# Define the output directory
output_dir = f"/home/niko/Documents/sicherung_trainings/multi_run_{current_date}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/src", exist_ok=True)

# Save important .py files
subprocess.run(["cp", "train_kd.py", f"{output_dir}/src"])
subprocess.run(["cp", "./visual_utils/vis_feature_maps.py", f"{output_dir}/src"])
subprocess.run(["cp", "../pcdet/models/backbones_3d/spconv_backbone.py", f"{output_dir}/src"])

# Loop over the parameters
for num_bins in num_bins_values:
    # add a subdirectory for each run that is named after the parameter
    output_dir_num_bins = f"{output_dir}/num_bins_{num_bins}"
    # Define the command as a list
    cmd = ["python", "train_kd.py", "--num_bins", str(num_bins), "--output_dir", output_dir_num_bins]
    print("Running with: ", cmd)
    # Run the command
    subprocess.run(cmd)