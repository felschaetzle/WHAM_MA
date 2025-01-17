import pickle
import joblib
import numpy as np
import torch

import argparse
from glob import glob
import os
import os.path as osp

from custom_utils import open_pkl, get_sequence_root

from lib.utils.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from lib.eval.eval_utils import compute_pred_trans_hat, global_align_joints, first_align_joints, align_pcl, align_extrinsics

from configs import constants as _C

from scipy.spatial.transform import Rotation as R
from lib.utils import transforms


WHAM_OUTPUT = "output/emdb"

def align_via_procrustes_torch(ground_truth, prediction):
    """
    Aligns the predicted points to the ground truth using Procrustes Analysis with scaling.
    
    Parameters:
        ground_truth (torch.Tensor): Ground truth points, shape (N, 3).
        prediction (torch.Tensor): Predicted points, shape (N, 3).
    
    Returns:
        aligned_prediction (torch.Tensor): Aligned prediction points, shape (N, 3).
        rotation_matrix (torch.Tensor): Optimal rotation matrix, shape (3, 3).
        translation_vector (torch.Tensor): Optimal translation vector, shape (3,).
        scale (float): Optimal scale factor.
    """
    # Ensure input is of type torch.float32
    ground_truth = ground_truth.float()
    prediction = prediction.float()
    
    # Center the points
    gt_mean = torch.mean(ground_truth, dim=0)
    pred_mean = torch.mean(prediction, dim=0)
    
    gt_centered = ground_truth - gt_mean
    pred_centered = prediction - pred_mean
    
    # Compute cross-covariance matrix
    H = pred_centered.T @ gt_centered
    
    # Perform SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation matrix
    R = U @ Vt
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Compute scale
    scale = torch.sum(S) / torch.sum(pred_centered.pow(2))
    
    # Compute translation vector
    t = gt_mean - scale * (R @ pred_mean)
    
    # Align the prediction
    aligned_prediction = scale * (R @ prediction.T).T + t
    
    return aligned_prediction, R, t, scale

def run(gt_pth, wham_pth, slam_pth, output_pth, args):
    yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float()

    annot = open_pkl(gt_pth)
    masks = annot['good_frames_mask']#[:850]
    gt_trans_world = annot["smpl"]["trans"]#[:850, :][masks]
    gt_pose_world = annot["smpl"]["poses_root"]#[:850, :][masks]
    gt_cam = annot["camera"]["extrinsics"]#[:850, :][masks]

    wham = open_pkl(wham_pth)
    wham = wham[0]
    pred_trans_world = wham["trans_world"]
    pred_pose_world = wham["pose_world"][:, :3]

    pred_pose_world = R.from_rotvec(pred_pose_world).as_matrix()
    slam_output = open_pkl(slam_pth)

    pred_cam_pose_orientation = R.from_quat(slam_output[:,3:]).as_matrix()
    pred_cam_pose_trans = slam_output[:,:3]

    gt_cam = np.linalg.inv(gt_cam)
    
    gt_trans_world = torch.from_numpy(gt_trans_world)
    pred_trans_world = torch.from_numpy(pred_trans_world)

    gt_pose_world = torch.from_numpy(gt_pose_world).unsqueeze(0).float()
    pred_pose_world = torch.from_numpy(pred_pose_world).unsqueeze(0).float()

    gt_cam = torch.from_numpy(gt_cam).float()
    pred_cam_pose_trans = torch.from_numpy(pred_cam_pose_trans).float()
    
    trans_hat, rot = compute_pred_trans_hat(gt_trans_world, pred_trans_world)
    root_poses_hat = rot @ pred_pose_world
    # root_poses_hat = yup2ydown @ rot @ pred_pose_world
    root_poses_hat = R.from_matrix(root_poses_hat.squeeze(0).numpy()).as_rotvec()

    # Align the gt cam and slam cam

    if args.gt_extrinsics:
        pred_cam_pose = gt_cam
    else:
        aligned_cam_pose, cam_pose_rot = compute_pred_trans_hat(gt_cam[:,:3,3], pred_cam_pose_trans)
        pred_cam_pose_orientation = cam_pose_rot @ pred_cam_pose_orientation
        # create pred_extrinsic matrix with same shape as gt_cam
        pred_cam_pose = np.zeros_like(gt_cam)
        pred_cam_pose[:, :3, :3] = pred_cam_pose_orientation
        pred_cam_pose[:, :3, 3] = aligned_cam_pose
        pred_cam_pose[:, 3, 3] = 1

    wham = {0: wham}
    wham[0]["trans_world_hat"] = trans_hat
    wham[0]["pose_world_hat"] = root_poses_hat
    wham[0]["cam_pose_hat"] = pred_cam_pose
    joblib.dump(wham, output_pth)

    print("DONE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=str, default=_C.subject_id, help="The subject ID, P0 - P9.")

    parser.add_argument(
        "--sequence",
        type=str,
        default=_C.sequence_id,
        help="The sequence ID. This can be any unambiguous prefix of the sequence's name, i.e. for the "
        "sequence '66_outdoor_rom' it could be '66' or any longer prefix including the full name.",
    )
    parser.add_argument("--gt_extrinsics", action='store_true', help="Use ground truth camera pose")

    parser.add_argument("--calib", help="Use ground truth camera pose")

    args = parser.parse_args()
    
    sequence_root = get_sequence_root(args, gt=True)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]

    sequence_root = get_sequence_root(args, gt=False)
    if args.gt_extrinsics:
        wham_data_path = glob(os.path.join(sequence_root, "*_output_gt_camera.pkl"))[0]
        slam_path = glob(os.path.join(sequence_root, "slam_results.pth"))[0] # unused but argument is required

    elif args.calib is not None:
        wham_data_path = glob(os.path.join(sequence_root, "*_output_gt_intrinsics.pkl"))[0]
        slam_path = glob(os.path.join(sequence_root, "slam_results_gt_intrinsics.pth"))[0]

    else:
        wham_data_path = glob(os.path.join(sequence_root, "*_output_DPVO.pkl"))[0]
        slam_path = glob(os.path.join(sequence_root, "slam_results.pth"))[0]

    # Output folder
    if args.gt_extrinsics:
        sequence = "wham_output_gt_camera_processed.pkl"
    elif args.calib is not None:
        sequence = "wham_output_gt_intrinsics_processed.pkl"
    else:
        sequence = "wham_output_DPVO_processed.pkl"
    output_pth = osp.join(sequence_root, sequence)
    print(output_pth)
    run(gt_data_path, wham_data_path, slam_path,  output_pth, args)
