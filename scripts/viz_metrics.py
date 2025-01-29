import joblib


emdb = joblib.load("dataset/parsed_data/emdb_2_vit.pth")
emdb_seq = emdb["vid"]
print(len(emdb_seq), "sequences in EMDB 2")
rts_sequences_gt_intrinsics = []
rts_sequences = []
seqs = []
for elm in emdb_seq:
    # cut of string at the second underscore
    a = elm.split("_", 2)
    a = a[0] + "_" + a[1]
    try:
        wham_data_gt_intrinsics = joblib.load("output/emdb/" + a + "/wham_output_gt_intrinsics_processed.pkl")
        rte_gt_intrinsics = wham_data_gt_intrinsics[0]["rte"]
        rts_sequences_gt_intrinsics.append(rte_gt_intrinsics)
        wham_data = joblib.load("output/emdb/" + a + "/wham_output_DPVO_processed.pkl")
        rte = wham_data[0]["rte"]
        rts_sequences.append(rte)
        seqs.append(a)
    except:
        print(a, " not found")

print(len(rts_sequences), "sequences found")

# plot rts and rts gt_intrinsics next to each other in a bar graph
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
barWidth = 0.3
r1 = np.arange(len(rts_sequences))
r2 = [x + barWidth for x in r1]
plt.bar(r1, rts_sequences, color="b", width=barWidth, edgecolor="grey", label="DPVO")
plt.bar(r2, rts_sequences_gt_intrinsics, color="r", width=barWidth, edgecolor="grey", label="gt_intrinsics")
plt.xlabel("Sequence", fontweight="bold")
plt.xticks([r + barWidth for r in range(len(rts_sequences))], seqs)
plt.ylabel("RTE")
plt.legend()
plt.show()
