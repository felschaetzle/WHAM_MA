import os
import subprocess
from glob import glob

# Set the path to the root directory of your dataset
DATASET_DIR = "/mnt/hdd/emdb_dataset/"

# Paths to your scripts
SCRIPT1 = "./script1.sh"
SCRIPT2 = "./script2.sh"

def execute_script(script, sub_id, seq_id, cam="DPVO"):
	try:
		if cam == "DPVO":
			command = [
				"python", 
				script, 
				"--subject",
				sub_id,
				"--sequence",
				seq_id
			]
		elif cam == "gt_intrinsics":
			command = [
				"python", 
				script, 
				"--subject",
				sub_id,
				"--sequence",
				seq_id,

				"--calib",
				"output/emdb/" + sub_id + "_" + seq_id[:2] + "/gt_intrinsics.txt"
			]
		elif cam == "gt_extrinsics":
			command = [
				"python", 
				script, 
				"--subject",
				sub_id,
				"--sequence",
				seq_id,
				"--gt_extrinsics"
			]

		print(command)

		result = subprocess.run(command)

		# Print the output and error (if any)
		print("Output:\n", result.stdout)

	except subprocess.CalledProcessError as e:
		print(f"Error running {script}")
		print(e.stderr)

def main():
	subdirectories = glob(f"{DATASET_DIR}/*/*/")
	subdirectories = sorted(subdirectories)
	# print(sorted(subdirectories))
	for path in subdirectories:
		relative_path = os.path.relpath(path, DATASET_DIR)  # Get relative path
		parts = relative_path.split(os.sep) 
		if len(parts) == 2:  # Ensure it is two levels deep
			subject_id = parts[0]
			sequence_id = parts[1].split('_')[0]  # Extract sequence ID (e.g., "00_mvs_a" -> 0)
			print(subject_id, sequence_id)

			# execute_script("create_mov_file.py", subject_id, sequence_id)

			# execute_script("demo.py", subject_id, sequence_id, gt_camera=False)

			execute_script("demo.py", subject_id, sequence_id, cam="gt_intrinsics")

			# execute_script("align_emdb.py", subject_id, sequence_id, gt_camera=False)

			execute_script("align_emdb.py", subject_id, sequence_id, cam="gt_intrinsics")

		else:
			print("FUCK!")

if __name__ == "__main__":
	main()
