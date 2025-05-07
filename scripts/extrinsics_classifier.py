import joblib

import sys
from pathlib import Path

# Add the parent directory of 'lib' to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import parse_args
from scripts.custom_utils import get_sequence_root

sys.path.append("/home/felix/WHAM_MA/scripts")

from superglue_tracker import get_fundamental_matrix, compute_epipolar_lines_batch, epipolar_distances_batch
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


import numpy as np

def compute_frame_relatives(ext: np.ndarray) -> np.ndarray:
    """
    ext: (N,4,4) absolute extrinsics
    returns rel: (N-1,4,4) where
      rel[i] = inv(ext[i]) @ ext[i+1]
    """
    N = ext.shape[0]
    rel = np.zeros((N-1, 4, 4), dtype=np.double)
    for i in range(N-1):
        rel[i] = ext[i+1] @ np.linalg.inv(ext[i]) 

    C = np.linalg.inv(ext)[:, :3, 3]
    t = np.linalg.norm(rel[:, :3, 3], axis=-1)
    dC = np.linalg.norm(C[1:] - C[:-1], axis=-1)
    # import ipdb; ipdb.set_trace()
    # assert np.allclose(t, dC), "the t should be of the same scale"
    return rel

def stitch_with_relatives(windows: list,
                          decisions: list,
                          dpvo_ext: np.ndarray,
                          wham_ext: np.ndarray,
                          tracker_Rt=None):
    """
    windows:   List of (start, end) frame‐ranges, [start,end) covering [0..N)
    decisions: List of bool, same length as windows:
               True → use DPVO in that window, False → WHAM.
    dpvo_ext:  (N,4,4) per‐frame DPVO absolutes
    wham_ext:  (N,4,4) per‐frame WHAM absolutes

    returns global_ext: (N,4,4) stitched absolutes
    """

    N = dpvo_ext.shape[0]
    assert wham_ext.shape[0] == N
    assert len(windows) == len(decisions)

    # 1) Precompute the per‐frame deltas for each pipeline
    rel_dpvo = compute_frame_relatives(dpvo_ext)   # shape (N-1,4,4)
    rel_wham = compute_frame_relatives(wham_ext)

    # 2) Build a bool array per‐relative‐transform telling us which pipeline to use
    chosen_dpvo = np.zeros(N-1, dtype=bool)
    for (s,e), use_dpvo in zip(windows, decisions):
        # assign every i where s <= i < e  → belongs to this window
        # those deltas will use the window’s decision
        start = s
        end   = min(e, N)  # e can be N only for the last window
        chosen_dpvo[start:end] = use_dpvo

    # 3) Integrate
    global_ext = np.zeros((N,4,4), dtype=np.float64)

    # initialize frame 0 with whichever pipeline you chose for window 0
    init_dpvo = decisions[0]
    global_ext[0] = dpvo_ext[0] if init_dpvo else wham_ext[0]

    # tracker_ext = global_ext.copy()

    # import ipdb; ipdb.set_trace()
    for i in range(N-1):
        if i == 30:
            print("h")
        Trel = rel_dpvo[i] if chosen_dpvo[i] else rel_wham[i]

        # chain it
        global_ext[i+1] = Trel @ global_ext[i]

        # tracker_ext[i+1] = tracker_Rt[i] @ tracker_ext[i]
    # import ipdb; ipdb.set_trace()

    return global_ext

def compute_relative_RT_from_tracks(kp0, kp1, K,
                                    conf=None, conf_thresh=0.5,
                                    method=cv2.RANSAC, prob=0.999,
                                    threshold=1.0):
    """
    Estimate relative pose (R, t) between two frames from tracked pixels.

    Args:
        kp0        (M×2 array): pixel coords in frame 0
        kp1        (M×2 array): corresponding pixel coords in frame 1
        K          (3×3 array): camera intrinsic matrix
        conf       (M,)       : optional match confidences
        conf_thresh float      : only keep matches with conf > conf_thresh
        method     int         : cv2.FIND_... (RANSAC or LMEDS)
        prob       float       : RANSAC success probability
        threshold  float       : inlier threshold (px)

    Returns:
        R (3×3 array): rotation from frame 0 → frame 1
        t (3×1 array): translation (unit length, up to scale)
        mask (N,)     : boolean mask of inlier matches
    """
    # optionally filter by confidence
    if conf is not None:
        keep = (conf > conf_thresh)
        kp0_f = kp0[keep]
        kp1_f = kp1[keep]
    else:
        kp0_f = kp0
        kp1_f = kp1

    if kp0_f.shape[0] < 5:
        raise ValueError(f"Need ≥5 matches, got {kp0_f.shape[0]}")

    # 1) Estimate Essential matrix E
    E, inlier_mask = cv2.findEssentialMat(
        kp0_f, kp1_f, K,
        method=method,
        prob=prob,
        threshold=threshold
    )
    # 2) Recover pose: R, t (t is unit norm, scale unknown)
    _, R, t, pose_mask = cv2.recoverPose(E, kp0_f, kp1_f, K, mask=inlier_mask)

    # combine masks: only those marked as inliers by both steps
    final_mask = (inlier_mask.ravel() > 0) & (pose_mask.ravel() > 0)

    return R, t, final_mask

def run_test(args):
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

    length = frame_mask.sum()
    # length = 300
    frames = np.array(range(len(image_paths)))[frame_mask]
    
    dpvo_ext = dpvo_ext[frames]

    keyframe_id = 0
    keyframe_id_wham = 0
    decision_use_dpvo = []
    windows = []
    tracker_Rt = []
    
    for i, elm in tqdm(enumerate(frames), total=frames.shape[0]):
        # No matches for frame 0
        if i == 0:
            continue

        if i == length-1 or tracks_db[i+1]["key_frame"] != keyframe_id:
            tracks = tracks_db[i]

            keyframe_id = tracks['key_frame']
            keyframe_id_wham = int(np.where(frames == keyframe_id)[0])
    
            kp0 = tracks['kp0']
            kp1 = tracks['kp1']
            conf = tracks['conf']

            if kp0 is not None:
                gt_F = get_fundamental_matrix(K, gt_ext[keyframe_id], gt_ext[elm])
                gt_lines = compute_epipolar_lines_batch(gt_F, kp0)
                gt_epi_error = epipolar_distances_batch(gt_lines, kp1)

                dpvo_F = get_fundamental_matrix(K, dpvo_ext[keyframe_id_wham], dpvo_ext[i])
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

                if (dpvo_epi_error.mean() < wham_epi_error.mean() and 
                (tracks['current_frame'] - tracks['key_frame']) > 10 and 
                dpvo_epi_error.mean() < 100):
                    decision_use_dpvo.append(True)
                else:
                    decision_use_dpvo.append(False)

                R_rel, t_rel, m = compute_relative_RT_from_tracks(kp0, kp1, K, conf)
                T_tracker = np.eye(4)
                T_tracker[:3,:3] = R_rel
                T_tracker[:3, 3] = t_rel.reshape((3,))
                tracker_Rt.append(T_tracker)
            else:
                decision_use_dpvo.append(False)
                wham_epi_error_list.append(0)
                dpvo_epi_error_list.append(0)
                gt_epi_error_list.append(0)

            index.append(elm)
            if i < length - 1:
                windows.append((keyframe_id_wham, i))
            else:
                windows.append((keyframe_id_wham, i))

            if i < length - 1:
                keyframe_id = tracks_db[i+1]['key_frame']
                keyframe_id_wham = int(np.where(frames == keyframe_id)[0])


    print(f"{sum(decision_use_dpvo)/len(decision_use_dpvo)}% of {len(decision_use_dpvo)} windows use DPVO")
    # print(windows)
    camera_init = stitch_with_relatives(windows, decision_use_dpvo, dpvo_ext, wham_ext, gt_ext, tracker_Rt)

    baseline['camera_extinsics_init'] = camera_init
    # baseline['tracker_extrinsics'] = tracker_ext

    joblib.dump(baseline, wham_root+'/baseline_gt_betas.pkl')

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

def extrinsic_classifier(args, wham_ext, dpvo_ext):
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
    K = gt_data['camera']['intrinsics'] 
    frame_mask = gt_data['good_frames_mask']

    index = []

    conf = []

    length = frame_mask.sum()
    # length = 300
    frames = np.array(range(len(image_paths)))[frame_mask]
    
    if dpvo_ext.shape[0] != frame_mask.sum():
        dpvo_ext = dpvo_ext[frames]

    keyframe_id = 0
    keyframe_id_wham = 0
    decision_use_dpvo = []
    windows = []
    tracker_Rt = []
    track_kp0 = []
    track_kp1 = []
    
    for i, elm in tqdm(enumerate(frames), total=frames.shape[0]):
        # No matches for frame 0
        if i == 0:
            continue

        if i == length-1 or tracks_db[i+1]["key_frame"] != keyframe_id:
            tracks = tracks_db[i]

            keyframe_id = tracks['key_frame']
            keyframe_id_wham = int(np.where(frames == keyframe_id)[0])
    
            kp0 = tracks['kp0']
            kp1 = tracks['kp1']
            conf = tracks['conf']

            if kp0 is not None:
                dpvo_F = get_fundamental_matrix(K, dpvo_ext[keyframe_id_wham], dpvo_ext[i])
                dpvo_lines = compute_epipolar_lines_batch(dpvo_F, kp0)
                dpvo_epi_error = epipolar_distances_batch(dpvo_lines, kp1)

                wham_F = get_fundamental_matrix(K, wham_ext[keyframe_id_wham], wham_ext[i])
                wham_lines = compute_epipolar_lines_batch(wham_F, kp0)
                wham_epi_error = epipolar_distances_batch(wham_lines, kp1)

                percentile = 50

                dpvo_thresh   = np.percentile(dpvo_epi_error, percentile)                            # median
                dpvo_keep_np  = dpvo_epi_error[dpvo_epi_error <= dpvo_thresh]

                wham_thresh   = np.percentile(wham_epi_error, percentile)                            # median
                wham_keep_np  = wham_epi_error[wham_epi_error <= wham_thresh]

                if (dpvo_keep_np.mean() < wham_keep_np.mean() and 
                (tracks['current_frame'] - tracks['key_frame']) > 15 and 
                dpvo_keep_np.mean() < 10):
                    decision_use_dpvo.append(True)
                else:
                    decision_use_dpvo.append(False)

                # R_rel, t_rel, m = compute_relative_RT_from_tracks(kp0, kp1, K, conf)
                # T_tracker = np.eye(4)
                # T_tracker[:3,:3] = R_rel
                # T_tracker[:3, 3] = t_rel.reshape((3,))
                # tracker_Rt.append(T_tracker)
            else:
                decision_use_dpvo.append(False)

            if i < length - 1:
                windows.append((keyframe_id_wham, i))
                track_kp0.append(kp0)
                track_kp1.append(kp1)
            else:
                windows.append((keyframe_id_wham, i))
                track_kp0.append(kp0)
                track_kp1.append(kp1)

            if i < length - 1:
                keyframe_id = tracks_db[i+1]['key_frame']
                keyframe_id_wham = int(np.where(frames == keyframe_id)[0])


    print(f"{(sum(decision_use_dpvo)/len(decision_use_dpvo))*100}% of {len(decision_use_dpvo)} windows use DPVO")
    # print(windows)
    camera_init = stitch_with_relatives(windows, decision_use_dpvo, dpvo_ext, wham_ext)

    return camera_init, windows, track_kp0, track_kp1

if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True) 
    run_test(args)