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
        
        files = {}

        # wham_path = f"output/emdb2/{a}/eval.pkl"
        # files["WHAM"] = wham_path
        # smplify_path = f"output/emdb2/{a}/smplify.pkl"
        # files["SMPLify"] = smplify_path
        # smplify_naive_path = f"output/emdb2/{a}/smplify_naive_intrinsics.pkl"
        # files["SMPLify naive intrinsics"] = smplify_naive_path
        # baseline_path = f"output/emdb2/{a}/baseline.pkl"
        # files["Baseline"] = baseline_path
        # baseline_beta_path = f"output/emdb2/{a}/baseline_gt_betas.pkl"
        # files["Baseline gt betas"] = baseline_beta_path        
        baseline_beta_at_once_path = f"output/emdb2/{a}/baseline_gt_betas_at_once.pkl"
        files["Baseline gt betas at once"] = baseline_beta_at_once_path

        # Store data in dictionary
        all_data[a] = {}

        for file, path in files.items():
            if os.path.exists(path):
                wham_data = joblib.load(path)
                all_data[a][file] = {
                    "rte": wham_data["rte"],
                    # "pa_mpjpe": wham_data["pa_mpjpe"],
                    # "mpjpe": wham_data["mpjpe"],
                    "w_mpjpe": wham_data["w_mpjpe"],
                    "wa_mpjpe": wham_data["wa_mpjpe"],
                }

    # Save all collected data into a single file
    output_file = "output/emdb2/combined_metrics_at_once.pkl"
    joblib.dump(all_data, output_file)

    print(f"Data saved successfully to {output_file}")


# # -------------------------------
# export all_data a csv file that I can use to create a table in latex
# # -------------------------------

import pandas as pd
import numpy as np

data_list = []
for seq, files_dict in all_data.items():
    row = {}
    for file_type, metrics in files_dict.items():
        for metric, value in metrics.items():
            # Create a column name by combining file type and metric name
            col_name = f"{file_type}_{metric}"
            row[col_name] = np.round(value, 3)
    row['sequence'] = seq  # Save the sequence name
    data_list.append(row)

# Convert list of dicts to DataFrame and set sequence as the index
df = pd.DataFrame(data_list).set_index('sequence')

# for each column, calculate the mean and add it to the dataframe
mean_row = df.mean(axis=0)
mean_row.name = 'Mean'
df = pd.concat([df, pd.DataFrame([mean_row])])


# Save the DataFrame to a csv file
output_file = "output/emdb2/combined_metrics_at_once.csv"
df.astype(float)
df.to_csv(output_file, sep=',', decimal='.')
print(f"Data saved successfully to {output_file}")
print(df)