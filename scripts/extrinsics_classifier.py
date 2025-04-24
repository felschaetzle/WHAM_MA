import joblib

import sys
from pathlib import Path

# Add the parent directory of 'lib' to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import parse_args
from scripts.custom_utils import get_sequence_root
from superglue_tracker import get_fundamental_matrix, compute_epipolar_lines_batch, epipolar_distances_batch
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def run(args):
    tracks_db = joblib.load(f"output/tracker/{args.sequence}.pkl")
    dumy_list = [0]
    dumy_list.extend(tracks_db)
    tracks_db = dumy_list.copy()

    root = get_sequence_root(args, gt=True)
    image_dir = Path(root) / "images"  # Adjust this path as needed
     # replace with your actual path
    image_paths = sorted(image_dir.glob("*.jpg"))  # or *.jpg if needed
    assert len(image_paths) >= 2, "Need at least 2 images to match"

    gt_data_pth = glob(os.path.join(root,"*.pkl"))[0]
    
    print(f"Process sequence: {args.sequence}")

    gt_data = joblib.load(gt_data_pth)
    bbox = gt_data['bboxes']['bboxes']
    gt_ext = gt_data['camera']['extrinsics']
    K = gt_data['camera']['intrinsics'] 
    frame_mask = gt_data['good_frames_mask']

    wham_root = get_sequence_root(args, False)
 
    baseline = joblib.load(wham_root+'/baseline_gt_betas.pkl')
    dpvo_ext = baseline['dpvo_extrinsics']

    if args.smooth_wham_cam:
        wham_ext = baseline['wham_cam']
    else:
        wham_ext = baseline['wham_cam_init']

    gt_epi_error_list = []
    dpvo_epi_error_list = []
    wham_epi_error_list = []
    index = []

    num_keypoints = []
    conf = []

    lenght = frame_mask.sum()
    frames = np.array(range(len(image_paths)))[frame_mask]

    switch_keyframe = True
    keyframe = 0
    keyframe_id = 0
    keyframe_id_wham = 0
    
    for i, elm in tqdm(enumerate(frames), total=frames.shape[0]):
        # No matches for frame 0
        if i == 0:
            continue
        

        if i == lenght-1 or tracks_db[i+1]["kp0"].shape[0] < 10:
            tracks = tracks_db[i]

            keyframe_id = tracks['key_frame']
            keyframe_id_wham = int(np.where(frames == keyframe_id)[0])
    
            kp0 = tracks['kp0']
            kp1 = tracks['kp1']
            conf = tracks['conf']

            gt_F = get_fundamental_matrix(K, gt_ext[keyframe_id], gt_ext[elm])
            gt_lines = compute_epipolar_lines_batch(gt_F, kp0)
            gt_epi_error = epipolar_distances_batch(gt_lines, kp1)

            dpvo_F = get_fundamental_matrix(K, dpvo_ext[keyframe_id], dpvo_ext[elm])
            dpvo_lines = compute_epipolar_lines_batch(dpvo_F, kp0)
            dpvo_epi_error = epipolar_distances_batch(dpvo_lines, kp1)

            wham_F = get_fundamental_matrix(K, wham_ext[keyframe_id_wham], wham_ext[i])
            wham_lines = compute_epipolar_lines_batch(wham_F, kp0)
            wham_epi_error = epipolar_distances_batch(wham_lines, kp1)

            percentile = 50

            gt_thresh   = np.percentile(gt_epi_error, percentile)                            # median
            gt_keep_np  = gt_epi_error[gt_epi_error <= gt_thresh]
            gt_epi_error_list.append(gt_keep_np.mean())

            dpvo_thresh   = np.percentile(dpvo_epi_error, percentile)                            # median
            dpvo_keep_np  = dpvo_epi_error[dpvo_epi_error <= dpvo_thresh]
            dpvo_epi_error_list.append(dpvo_keep_np.mean())

            wham_thresh   = np.percentile(wham_epi_error, percentile)                            # median
            wham_keep_np  = wham_epi_error[wham_epi_error <= wham_thresh]
            wham_epi_error_list.append(wham_keep_np.mean())

            index.append(elm)
            # print(f"GT {gt_epi_error.mean()}, DVPO {dpvo_epi_error.mean()}, WHAM {wham_epi_error.mean()}")

        else:
            gt_epi_error_list.append(np.nan)
            dpvo_epi_error_list.append(np.nan)
            wham_epi_error_list.append(np.nan)
            index.append(elm)

            # print(f"SKIP {elm}, not enough matches.")

    # === Plot 1: Epipolar Errors === #
    p = f"output/tracker/{args.sequence}_epipolor_distance.png"
    plt.figure(figsize=(10, 5))
    plt.plot(index, gt_epi_error_list, label='GT', marker='o')
    plt.plot(index, dpvo_epi_error_list, label='DPVO', marker='o')
    plt.plot(index, wham_epi_error_list, label='WHAM', marker='o')

    plt.xlabel('Frame Index')
    plt.ylabel('Epipolar Error (pixels)')
    plt.title('Epipolar Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p)
    print("Epipolar error plot saved to", p)

    print(f"Mean WHAM {np.nanmean(wham_epi_error_list)}, DPVO {np.nanmean(dpvo_epi_error_list)}")

if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True)
    run(args)