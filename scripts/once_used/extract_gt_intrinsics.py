import os
import subprocess
from glob import glob
import joblib
import argparse
import numpy as np

# Set the path to the root directory of your dataset
DATASET_DIR = "/mnt/hdd/emdb_dataset/"
TARGET_ROOT = "output/emdb2"


def main():
	emdb = joblib.load("dataset/parsed_data/emdb_2_vit.pth")
	emdb_seq = emdb["vid"]
	for elm in emdb_seq:
		a = "_".join(elm.split("_", 2)[:2])  # Extract sequence name
		sub = a.split("_")[0]
		path = DATASET_DIR + "/" + sub + "/" + elm[3:] + "/" + elm + "_data.pkl"
		target_path = TARGET_ROOT + "/" + a + "/gt_intrinsics.txt"
		data = joblib.load(path)
		intrinsics = data["camera"]["intrinsics"]
		wham_format = [np.round(intrinsics[0, 0],2), np.round(intrinsics[1, 1],2), np.round(intrinsics[0, 2],2), np.round(intrinsics[1, 2],2)]
		with open(target_path, "w") as f:
			f.write(" ".join(map(str, wham_format)))
		print(target_path, "written")
if __name__ == "__main__":
	main()
