import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the root directory
path = 'output/tracker'

# Fix the glob pattern and get all CSV files recursively
csv_list = glob(os.path.join(path, '*.csv'))
csv_list = sorted(csv_list)

# Prepare data for plotting
num_keypoints = []
seq = []

for csv_file in csv_list:
    df = pd.read_csv(csv_file)
    kps = df['num_keypoints'].mean()
    num_keypoints.append(kps)
    seq.append(csv_file[-6:-4])
# Plot num_keypoints for each sequence

plt.figure(figsize=(12, 6))
plt.plot(seq, num_keypoints, label="Number Keypoints")

plt.xlabel("Sequence")
plt.ylabel("Number of Keypoints")
plt.title(f"Mean number of keypoints per Sequence. Total mean: {np.mean(num_keypoints)}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/tracker/keypoints_per_sequence.png")
