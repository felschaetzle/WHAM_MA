import os
import subprocess
from glob import glob

# Set the path to the root directory of your dataset
DATASET_DIR = "output/emdb"

# Paths to your scripts
SCRIPT1 = "./script1.sh"
SCRIPT2 = "./script2.sh"

def execute_script(script, sub_id=None, seq_id=None):
	try:
		command = [
			"python", 
			script, 
			"--subject",
			sub_id,
			"--sequence",
			seq_id
		]
		print(command)

		result = subprocess.run(command)

		# Print the output and error (if any)
		print("Output:\n", result.stdout)

	except subprocess.CalledProcessError as e:
		print(f"Error running {script}")
		print(e.stderr)

def main():
	subdirectories = glob(f"{DATASET_DIR}/*/wham_output_full_gt_camera_processed.pkl")
	subdirectories = sorted(subdirectories)
	for old_path in subdirectories:
		relative_path = os.path.relpath(old_path, DATASET_DIR)  # Get relative path
		parts = relative_path.split(os.sep)
		new_name = "wham_output_DPVO_processed.pkl"
		new_name = os.path.join(DATASET_DIR, parts[0], new_name)
		print(old_path, "to: ", new_name)
		# os.rename(old_path, new_name)
		os.remove(old_path)

if __name__ == "__main__":
	main()
