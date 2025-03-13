# import joblib
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the combined data
# combined_data_path = "output/emdb2/combined_metrics.pkl"
# all_data = joblib.load(combined_data_path)

# # Extract sequences and RTE values for DPVO vs gt_intrinsics
# sequences = list(all_data.keys())
# rte_dpvo = [all_data[seq]["DPVO"]["rte"] if "DPVO" in all_data[seq] else np.nan for seq in sequences]
# rte_gt_intrinsics = [all_data[seq]["gt_intrinsics"]["rte"] if "gt_intrinsics" in all_data[seq] else np.nan for seq in sequences]

# # Calculate average RTE values
# avg_rte_dpvo = np.nanmean(rte_dpvo)
# avg_rte_gt_intrinsics = np.nanmean(rte_gt_intrinsics)

# # Plot DPVO vs gt_intrinsics
# fig, ax = plt.subplots(figsize=(12, 6))
# bar_width = 0.3
# r1 = np.arange(len(sequences))
# r2 = [x + bar_width for x in r1]

# plt.bar(r1, rte_dpvo, color="b", width=bar_width, edgecolor="grey", label="DPVO")
# plt.bar(r2, rte_gt_intrinsics, color="r", width=bar_width, edgecolor="grey", label="gt_intrinsics")

# plt.xlabel("Sequence", fontweight="bold")
# plt.xticks([r + bar_width/2 for r in range(len(sequences))], sequences, rotation=90)
# plt.ylabel("RTE")
# plt.title(f"Comparison of RTE (Avg DPVO: {avg_rte_dpvo:.3f}, Avg gt_intrinsics: {avg_rte_gt_intrinsics:.3f})")
# plt.legend()

# # Extract sequences and RTE values for gt_intrinsics vs gt_camera (if available)
# rte_gt_extrinsics = [all_data[seq]["gt_camera"]["rte"] if "gt_camera" in all_data[seq] else np.nan for seq in sequences]

# # Calculate average RTE values
# avg_rte_gt_extrinsics = np.nanmean(rte_gt_extrinsics)

# # Plot gt_intrinsics vs gt_camera
# fig, ax = plt.subplots(figsize=(12, 6))

# plt.bar(r1, rte_gt_intrinsics, color="r", width=bar_width, edgecolor="grey", label="gt_intrinsics")
# plt.bar(r2, rte_gt_extrinsics, color="g", width=bar_width, edgecolor="grey", label="gt_camera")

# plt.xlabel("Sequence", fontweight="bold")
# plt.xticks([r + bar_width for r in range(len(sequences))], sequences, rotation=90)
# plt.ylabel("RTE")
# plt.title(f"Comparison of RTE (Avg gt_intrinsics: {avg_rte_gt_intrinsics:.3f}, Avg gt_camera: {avg_rte_gt_extrinsics:.3f})")
# plt.legend()
# plt.show()



import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data
combined_data_path = "output/emdb2/combined_metrics.pkl"
all_data = joblib.load(combined_data_path)

# Extract sequences
sequences = list(all_data.keys())

# Compute RTE values (computed only once)
rte_dpvo = [
    all_data[seq]["DPVO"]["rte"] if "DPVO" in all_data[seq] else np.nan 
    for seq in sequences
]
rte_gt_intrinsics = [
    all_data[seq]["gt_intrinsics"]["rte"] if "gt_intrinsics" in all_data[seq] else np.nan 
    for seq in sequences
]
rte_gt_camera = [
    all_data[seq]["gt_camera"]["rte"] if "gt_camera" in all_data[seq] else np.nan 
    for seq in sequences
]
rte_gt_camera_wo_SMPLify = [
    all_data[seq]["gt_camera_wo_SMPLify"]["rte"] if "gt_camera_wo_SMPLify" in all_data[seq] else np.nan 
    for seq in sequences
]

rte_gt_intrinsics_baseline = [
    all_data[seq]["gt_instrinsics_baseline"]["rte"] if "gt_instrinsics_baseline" in all_data[seq] else np.nan 
    for seq in sequences
]


# Calculate average RTE values
avg_rte_dpvo = np.nanmean(rte_dpvo)
avg_rte_gt_intrinsics = np.nanmean(rte_gt_intrinsics)
avg_rte_gt_camera = np.nanmean(rte_gt_camera)
avg_rte_gt_camera_wo_SMPLify = np.nanmean(rte_gt_camera_wo_SMPLify)
avg_rte_gt_intrinsics_baseline = np.nanmean(rte_gt_intrinsics_baseline)

# Common parameters for the bar plots
bar_width = 0.3
r1 = np.arange(len(sequences))
r2 = [x + bar_width for x in r1]

# -------------------------------
# Chart 1: DPVO vs gt_intrinsics
# -------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(r1, rte_dpvo, color="b", width=bar_width, edgecolor="grey", label="WHAM")
ax1.bar(r2, rte_gt_intrinsics, color="r", width=bar_width, edgecolor="grey", label="Adding GT intrinsics")
ax1.set_xlabel("Sequence", fontweight="bold")
ax1.set_xticks([r + bar_width/2 for r in range(len(sequences))])
ax1.set_xticklabels(sequences, rotation=90)
ax1.set_ylabel("RTE")
ax1.set_title(f"Comparison of RTE (Avg WHAM: {avg_rte_dpvo:.3f}, Avg adding GT intrinsics: {avg_rte_gt_intrinsics:.3f})")
ax1.legend()

# -------------------------------
# Chart 2: gt_intrinsics vs gt_camera
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.bar(r1, rte_gt_intrinsics, color="r", width=bar_width, edgecolor="grey", label="Adding GT intrinsics")
ax2.bar(r2, rte_gt_camera, color="g", width=bar_width, edgecolor="grey", label="Adding GT camera")
ax2.set_xlabel("Sequence", fontweight="bold")
ax2.set_xticks([r + bar_width/2 for r in range(len(sequences))])
ax2.set_xticklabels(sequences, rotation=90)
ax2.set_ylabel("RTE")
ax2.set_title(f"Comparison of RTE (Avg GT intrinsics: {avg_rte_gt_intrinsics:.3f}, Avg GT camera: {avg_rte_gt_camera:.3f})")
ax2.legend()

# -------------------------------
# Chart 3: gt_camera vs gt_camera_wo_SMPLify
# -------------------------------
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.bar(r1, rte_gt_camera, color="g", width=bar_width, edgecolor="grey", label="Adding GT camera")
ax3.bar(r2, rte_gt_camera_wo_SMPLify, color="m", width=bar_width, edgecolor="grey", label="Adding GT camera wo SMPLify")
ax3.set_xlabel("Sequence", fontweight="bold")
ax3.set_xticks([r + bar_width/2 for r in range(len(sequences))])
ax3.set_xticklabels(sequences, rotation=90)
ax3.set_ylabel("RTE")
ax3.set_title(f"Comparison of RTE (Avg GT camera: {avg_rte_gt_camera:.3f}, Avg GT camera wo SMPLify: {avg_rte_gt_camera_wo_SMPLify:.3f})")
ax3.legend()

# -------------------------------
# Chart 4: gt_intrinsics vs gt_intrinsics_baseline
# -------------------------------
fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.bar(r1, rte_gt_intrinsics, color="r", width=bar_width, edgecolor="grey", label="Adding GT intrinsics")
ax4.bar(r2, rte_gt_intrinsics_baseline, color="c", width=bar_width, edgecolor="grey", label="Baseline")
ax4.set_xlabel("Sequence", fontweight="bold")
ax4.set_xticks([r + bar_width/2 for r in range(len(sequences))])
ax4.set_xticklabels(sequences, rotation=90)
ax4.set_ylabel("RTE")
ax4.set_title(f"Comparison of RTE (Avg GT intrinsics [3.1]: {avg_rte_gt_intrinsics:.3f}, Avg GT intrinsics baseline [2.1]: {avg_rte_gt_intrinsics_baseline:.3f})")
ax4.legend()

# Display all charts
plt.show()
