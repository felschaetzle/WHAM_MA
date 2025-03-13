import os
import subprocess
from glob import glob
import joblib
from custom_utils import find_substring
# Set the path to the root directory of your dataset
DATASET_DIR = "/mnt/hdd/emdb_dataset/"


def execute_demo(sub_id, seq_id, gt_intrinsics=False, gt_extrinsics=False):
	try:
		command = [
			"python", 
			"demo.py", 
			"--subject",
			sub_id,
			"--sequence",
			seq_id,
			"--gt_intrinsics",
			str(gt_intrinsics),
			"--gt_extrinsics",
			str(gt_extrinsics)
			]

		# print(command)

		result = subprocess.run(command)

		# Print the output and error (if any)
		print("Output:\n", result.stdout)

	except subprocess.CalledProcessError as e:
		print(f"Error running demo.py")
		print(e.stderr)

def execute_align(sub_id, seq_id, gt_intrinsics=False, gt_extrinsics=False, base_line=False):
	try:
		command = [
			"python", 
			"scripts/align_emdb.py", 
			"--subject",
			sub_id,
			"--sequence",
			seq_id,
			"--gt_intrinsics",
			str(gt_intrinsics),
			"--gt_extrinsics",
			str(gt_extrinsics),
			# "--baseline",
			]

		# print(command)

		result = subprocess.run(command)

		# Print the output and error (if any)
		print("Output:\n", result.stdout)

	except subprocess.CalledProcessError as e:
		print(f"Error running align.py")
		print(e.stderr)

def main():
	subdirectories = glob(f"{DATASET_DIR}/*/*/")
	subdirectories = sorted(subdirectories)
	emdb2 = joblib.load('dataset/parsed_data/emdb_2_vit.pth')
	# print(sorted(subdirectories))
	for path in subdirectories:
		relative_path = os.path.relpath(path, DATASET_DIR)  # Get relative path
		parts = relative_path.split(os.sep) 
		if len(parts) == 2:  # Ensure it is two levels deep
			subject_id = parts[0]
			sequence_id = parts[1].split('_')[0]  # Extract sequence ID (e.g., "00_mvs_a" -> 0)
			seq = subject_id + "_" + sequence_id
			emdb2_sequence = emdb2['vid']
			if find_substring(seq, emdb2_sequence) is None:
				print(f"Skipping {subject_id} {sequence_id} because it is not in emdb2")	
				continue
			
			# if len(os.listdir("output/emdb2/"+subject_id+"_"+sequence_id)) == 9:
			# 	print("Skipping", subject_id, sequence_id, "because it is already processed")
			# 	continue

			print(subject_id, sequence_id)

			# execute_script("create_mov_file.py", subject_id, sequence_id)

			# execute_demo(subject_id, sequence_id, gt_intrinsics=False, gt_extrinsics=False)

			# execute_demo(subject_id, sequence_id, gt_intrinsics=True, gt_extrinsics=False)

			execute_demo(subject_id, sequence_id, gt_intrinsics=True, gt_extrinsics=True)

			# execute_align(subject_id, sequence_id, gt_intrinsics=False, gt_extrinsics=False)

			# execute_align(subject_id, sequence_id, gt_intrinsics=True, gt_extrinsics=False, base_line=True)

			execute_align(subject_id, sequence_id, gt_intrinsics=True, gt_extrinsics=True)



		else:
			print("FAIL!")

if __name__ == "__main__":
	main()
