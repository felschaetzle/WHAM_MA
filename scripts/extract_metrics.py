import joblib
import os

# List of sequences to process
emdb = joblib.load("dataset/parsed_data/emdb_2_vit.pth")
emdb_seq = emdb["vid"]

# Dictionary to store all results
all_data = {}

for elm in emdb_seq:
    a = "_".join(elm.split("_", 2)[:2])  # Extract sequence name
    print(f"Processing sequence {a}")
    # Define file paths
    dpvo_path = f"output/emdb2/{a}/wham_output_DPVO_processed.pkl"
    gt_intrinsics_path = f"output/emdb2/{a}/wham_output_gt_intrinsics_processed.pkl"
    gt_camera_path = f"output/emdb2/{a}/wham_output_gt_camera_processed.pkl"
    gt_camera_wo_SMPLify_path = f"output/emdb2/{a}/wham_output_gt_camera_wo_SMPLify_processed.pkl"


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

# Save all collected data into a single file
output_file = "output/emdb2/combined_metrics.pkl"
joblib.dump(all_data, output_file)

print(f"Data saved successfully to {output_file}")
