import joblib
import os

# List of sequences to process
emdb = joblib.load("dataset/parsed_data/emdb_2_vit.pth")
emdb_seq = emdb["vid"]

# Dictionary to store all results
all_data = {}


if True:
    for elm in emdb_seq:
        a = "_".join(elm.split("_", 2)[:2])  # Extract sequence name
        print(f"Processing sequence {a}")
        # Define file paths
        dpvo_path = f"output/emdb2/{a}/wham_output_DPVO_processed.pkl"
        gt_intrinsics_path = f"output/emdb2/{a}/wham_output_gt_intrinsics_processed.pkl"
        gt_camera_path = f"output/emdb2/{a}/wham_output_gt_camera_processed.pkl"
        gt_camera_wo_SMPLify_path = f"output/emdb2/{a}/wham_output_gt_camera_wo_SMPLify_processed.pkl"
        gt_intrinsics_baseline_path = f"output/emdb2/{a}/wham_output_gt_intrinsics_baseline.pkl"


        # Store data in dictionary
        all_data[a] = {}

        if os.path.exists(dpvo_path):
            wham_data = joblib.load(dpvo_path)
            all_data[a]["DPVO"] = {
                "rte": wham_data["rte"],
                "pa_mpjpe": wham_data["pa_mpjpe"],
                "mpjpe": wham_data["mpjpe"],
                "w_mpjpe": wham_data["w_mpjpe"],
                "wa_mpjpe": wham_data["wa_mpjpe"],
            }

        if os.path.exists(gt_intrinsics_path):
            wham_data_gt_intrinsics = joblib.load(gt_intrinsics_path)
            all_data[a]["gt_intrinsics"] = {
                "rte": wham_data_gt_intrinsics["rte"],
                "pa_mpjpe": wham_data_gt_intrinsics["pa_mpjpe"],
                "mpjpe": wham_data_gt_intrinsics["mpjpe"],
                "w_mpjpe": wham_data_gt_intrinsics["w_mpjpe"],
                "wa_mpjpe": wham_data_gt_intrinsics["wa_mpjpe"],
            }
        
        if os.path.exists(gt_camera_path):
            wham_data_gt_camera = joblib.load(gt_camera_path)
            all_data[a]["gt_camera"] = {
                "rte": wham_data_gt_camera["rte"],
                "pa_mpjpe": wham_data_gt_camera["pa_mpjpe"],
                "mpjpe": wham_data_gt_camera["mpjpe"],
                "w_mpjpe": wham_data_gt_camera["w_mpjpe"],
                "wa_mpjpe": wham_data_gt_camera["wa_mpjpe"],
            }

        if os.path.exists(gt_camera_wo_SMPLify_path):
            wham_data_gt_camera_wo_SMPLify = joblib.load(gt_camera_wo_SMPLify_path)
            all_data[a]["gt_camera_wo_SMPLify"] = {
                "rte": wham_data_gt_camera_wo_SMPLify["rte"],
                "pa_mpjpe": wham_data_gt_camera_wo_SMPLify["pa_mpjpe"],
                "mpjpe": wham_data_gt_camera_wo_SMPLify["mpjpe"],
                "w_mpjpe": wham_data_gt_camera_wo_SMPLify["w_mpjpe"],
                "wa_mpjpe": wham_data_gt_camera_wo_SMPLify["wa_mpjpe"],
            }


        if os.path.exists(gt_intrinsics_baseline_path):
            wham_data_gt_intrinsics_baseline = joblib.load(gt_intrinsics_baseline_path)
            all_data[a]["gt_instrinsics_baseline"] = {
                "rte": wham_data_gt_intrinsics_baseline["rte"],
                "pa_mpjpe": wham_data_gt_intrinsics_baseline["pa_mpjpe"],
                "mpjpe": wham_data_gt_intrinsics_baseline["mpjpe"],
                "w_mpjpe": wham_data_gt_intrinsics_baseline["w_mpjpe"],
                "wa_mpjpe": wham_data_gt_intrinsics_baseline["wa_mpjpe"],
            }

    # Save all collected data into a single file
    output_file = "output/emdb2/combined_metrics.pkl"
    joblib.dump(all_data, output_file)

    print(f"Data saved successfully to {output_file}")


# # -------------------------------
# export all_data a csv file that I can use to create a table in latex
# # -------------------------------

import pandas as pd
import numpy as np

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

# rte_gt_intrinsics_baseline = [
#     all_data[seq]["gt_instrinsics_baseline"]["rte"] if "gt_instrinsics_baseline" in all_data[seq] else np.nan 
#     for seq in sequences
# ]

# create wa_mpjpe for DPVO and baseline

wa_mpjpe_dpvo = [
    all_data[seq]["DPVO"]["wa_mpjpe"] if "DPVO" in all_data[seq] else np.nan 
    for seq in sequences
]

wa_mpjpe_gt_intrinsics = [
    all_data[seq]["gt_intrinsics"]["wa_mpjpe"] if "gt_intrinsics" in all_data[seq] else np.nan 
    for seq in sequences
]

wa_mpjpe_gt_camera_wo_SMPLify = [
    all_data[seq]["gt_camera_wo_SMPLify"]["wa_mpjpe"] if "gt_camera_wo_SMPLify" in all_data[seq] else np.nan 
    for seq in sequences
]

# wa_mpjpe_gt_intrinsics_baseline = [
#     all_data[seq]["gt_instrinsics_baseline"]["wa_mpjpe"] if "gt_instrinsics_baseline" in all_data[seq] else np.nan 
#     for seq in sequences
# ]


# Calculate average RTE values
avg_rte_dpvo = np.nanmean(rte_dpvo)
avg_rte_gt_intrinsics = np.nanmean(rte_gt_intrinsics)
avg_rte_gt_camera = np.nanmean(rte_gt_camera)
avg_rte_gt_camera_wo_SMPLify = np.nanmean(rte_gt_camera_wo_SMPLify)
# avg_rte_gt_intrinsics_baseline = np.nanmean(rte_gt_intrinsics_baseline)
avg_mpjpe_dpvo = np.nanmean(wa_mpjpe_dpvo)
avg_mpjpe_gt_intrinsics = np.nanmean(wa_mpjpe_gt_intrinsics)
avg_mpjpe_gt_camera_wo_SMPLify = np.nanmean(wa_mpjpe_gt_camera_wo_SMPLify)
# avg_mpjpe_gt_intrinsics_baseline = np.nanmean(wa_mpjpe_gt_intrinsics_baseline)

# append to lists
sequences.append("Mean")
rte_dpvo.append(avg_rte_dpvo)
rte_gt_intrinsics.append(avg_rte_gt_intrinsics)
rte_gt_camera.append(avg_rte_gt_camera)
rte_gt_camera_wo_SMPLify.append(avg_rte_gt_camera_wo_SMPLify)
# rte_gt_intrinsics_baseline.append(avg_rte_gt_intrinsics_baseline)
wa_mpjpe_dpvo.append(avg_mpjpe_dpvo)
wa_mpjpe_gt_intrinsics.append(avg_mpjpe_gt_intrinsics)
wa_mpjpe_gt_camera_wo_SMPLify.append(avg_mpjpe_gt_camera_wo_SMPLify)
# wa_mpjpe_gt_intrinsics_baseline.append(avg_mpjpe_gt_intrinsics_baseline)

# Create a DataFrame
df = pd.DataFrame({
    "Sequence": sequences,
    "RTE WHAM": rte_dpvo,
    "RTE GT Intrinsics": rte_gt_intrinsics,
    "RTE GT Camera": rte_gt_camera,
    "RTE GT Camera w/o SMPLify": rte_gt_camera_wo_SMPLify,
    # "RTE Baseline": rte_gt_intrinsics_baseline,
    "WA MPJPE WHAM": wa_mpjpe_dpvo,
    "WA MPJPE GT Intrinsics": wa_mpjpe_gt_intrinsics,
    "WA MPJPE GT Camera w/o SMPLify": wa_mpjpe_gt_camera_wo_SMPLify,
    # "WA MPJPE Baseline": wa_mpjpe_gt_intrinsics_baseline
})


print("Mean values:")
print(df.iloc[-1])

# Save the DataFrame to a CSV file
output_csv = "output/emdb2/combined_metrics.csv"
df.to_csv(output_csv, index=False)

print(f"Data saved successfully to {output_csv}")