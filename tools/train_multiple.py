import subprocess

# Define the parameters for multiple runs
num_bins_values = [5, 15, 45]
x_shift_values = [0.7]
multiplier_values = [10, 15, 20]
eval_after_epoch = False

# Loop over the parameters
for num_bins in num_bins_values:
    # Define the command as a list
    cmd = ["python", "train_kd.py", "--num_bins", str(num_bins)]
    print("Running with: ", cmd)
    # Run the command
    subprocess.run(cmd)