import os
import subprocess
from glob import glob
import joblib
import argparse
import numpy as np

# Set the path to the root directory of your dataset
DATASET_DIR = "/mnt/hdd/emdb_dataset/"
TARGET_ROOT = "output/emdb"


def main():
	# sequence = args.subject + "_" + args.sequence
	# output_pth = osp.join(args.output_pth, sequence)

	subdirectories = glob(f"{DATASET_DIR}/*/*/*_data.pkl")
	subdirectories = sorted(subdirectories)
	for elm in subdirectories:
		relative_path = os.path.relpath(elm, DATASET_DIR)  # Get relative path
		parts = relative_path.split(os.sep)
		target_path = TARGET_ROOT + "/" + parts[0] + "_" + parts[1][:2] + "/gt_intrinsics.txt"
		data = joblib.load(elm)
		intrinsics = data["camera"]["intrinsics"]
		wham_format = [np.round(intrinsics[0, 0],2), np.round(intrinsics[1, 1],2), np.round(intrinsics[0, 2],2), np.round(intrinsics[1, 2],2)]
		with open(target_path, "w") as f:
			f.write(" ".join(map(str, wham_format)))
		print(target_path, "written")
if __name__ == "__main__":
	main()
