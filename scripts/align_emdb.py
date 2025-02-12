import pickle
import joblib
import numpy as np
import torch
from smplx import SMPL
import sys
sys.path.append('/home/felix/WHAM_MA')
print(sys.path)
from lib.models import build_network, build_body_model
from configs.config import get_cfg

import argparse
from glob import glob
import os
import os.path as osp

from custom_utils import open_pkl, get_sequence_root

from lib.utils.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from lib.eval.eval_utils import compute_pred_trans_hat, global_align_joints, first_align_joints, align_pcl, compute_jpe, batch_align_by_pelvis, batch_compute_similarity_transform_torch


import sys
sys.path.append("/home/felix/WHAM_MA")
from configs import constants as _C

from scipy.spatial.transform import Rotation as R
from lib.utils import transforms


m2mm = 1e3
pelvis_idxs = [1, 2]

def run(gt_pth, wham_pth, slam_pth, output_pth, args, cfg):

    yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float()

    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE) 

    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    smpl = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE) for k in ['male', 'female', 'neutral']}

    #######################################################################################################
    # Prepare GT data #####################################################################################
    #######################################################################################################
    annot = open_pkl(gt_pth)
    masks = annot['good_frames_mask']#[:850]
    gt_trans_world = annot["smpl"]["trans"][masks]
    gt_pose_world = annot["smpl"]["poses_root"][masks]
    gt_body_pose = annot["smpl"]["poses_body"][masks]
    gt_betas = np.repeat(annot["smpl"]["betas"].reshape((1, -1)), repeats=sum(masks), axis=0)
    gt_cam = annot["camera"]["extrinsics"][masks]
    gender = annot['gender']

    poses_root_cam = transforms.matrix_to_axis_angle(tt(gt_cam[:, :3, :3]) @ transforms.axis_angle_to_matrix(tt(gt_pose_world)))

    target_cam = smpl[gender](body_pose=tt(gt_body_pose), global_orient=poses_root_cam, betas=tt(gt_betas))
    target_verts_cam = target_cam.vertices
    target_j3d_cam = target_cam.joints[:, :24]

    # Groundtruth global motion
    target_glob = smpl[gender](body_pose=tt(gt_body_pose), global_orient=tt(gt_pose_world), betas=tt(gt_betas), transl=tt(gt_trans_world))
    target_j3d_glob = target_glob.joints[:, :24]
    
    #######################################################################################################
    # Prepare WHAM prediction data#########################################################################
    #######################################################################################################
    wham = open_pkl(wham_pth)
    # wham = wham[0]
    pred_trans_world = wham["trans_world"]
    pred_pose_world = wham["pose_world"][:, :3]
    body_pose = wham["pose_world"][:, 3:]
    betas = wham["betas"]
    root_cam = wham["pose"][:, :3]

    pred_pose_world = R.from_rotvec(pred_pose_world).as_matrix()
    body_pose = np.reshape(body_pose, (-1, 23, 3))
    # transform to matrix representation

    body_pose = transforms.axis_angle_to_matrix(tt(body_pose))
    root_cam = transforms.axis_angle_to_matrix(tt(root_cam))

    # Predicted local motion
    pred_cam = smpl['neutral'](body_pose=body_pose, global_orient=root_cam.unsqueeze(1), betas=tt(betas), pose2rot=False)
    pred_verts_cam = pred_cam.vertices
    pred_j3d_cam = pred_cam.joints[:, :24]

    # Predicted global motion
    pred_glob = smpl['neutral'](body_pose=body_pose, global_orient=tt(pred_pose_world).unsqueeze(1), betas=tt(betas), transl=tt(pred_trans_world), pose2rot=False)
    pred_j3d_glob = pred_glob.joints[:, :24]


    #######################################################################################################
    # Prepare SLAM prediction data ########################################################################
    #######################################################################################################
    slam_output = open_pkl(slam_pth)
    slam_output = slam_output[masks]
    pred_cam_pose_orientation = R.from_quat(slam_output[:,3:]).as_matrix()
    pred_cam_pose_trans = slam_output[:,:3]

    gt_cam = np.linalg.inv(gt_cam)
    
    gt_trans_world = torch.from_numpy(gt_trans_world)
    pred_trans_world = torch.from_numpy(pred_trans_world)

    gt_pose_world = torch.from_numpy(gt_pose_world).unsqueeze(0).float()
    pred_pose_world = torch.from_numpy(pred_pose_world).unsqueeze(0).float()

    gt_cam = torch.from_numpy(gt_cam).float()
    pred_cam_pose_trans = torch.from_numpy(pred_cam_pose_trans).float()
    
    # <======= Evaluation on the local motion
    pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = batch_align_by_pelvis(
        [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam], pelvis_idxs
    )
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d_cam, target_j3d_cam)
    pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy() * m2mm
    mpjpe = torch.sqrt(((pred_j3d_cam - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy() * m2mm
    print("PA-MPJPE: ", pa_mpjpe.mean())
    print("MPJPE: ", mpjpe.mean())

    # <======= Evaluation on the global motion

    chunk_length = 100
    w_mpjpe, wa_mpjpe = [], []
    for start in range(0, masks.sum(), chunk_length):
        end = min(masks.sum(), start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()
        
        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)
        
        w_jpe = compute_jpe(target_j3d, w_j3d)
        wa_jpe = compute_jpe(target_j3d, wa_j3d)
        w_mpjpe.append(w_jpe)
        wa_mpjpe.append(wa_jpe)
    
    w_mpjpe = np.concatenate(w_mpjpe) * m2mm
    wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm

    print("W-MPJPE: ", w_mpjpe.mean())
    print("WA-MPJPE: ", wa_mpjpe.mean())

    trans_hat, rot = compute_pred_trans_hat(gt_trans_world, pred_trans_world)
    root_poses_hat = rot @ pred_pose_world
    # root_poses_hat = yup2ydown @ rot @ pred_pose_world
    root_poses_hat = R.from_matrix(root_poses_hat.squeeze(0).numpy()).as_rotvec()


    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(gt_trans_world, gt_trans_world[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)
    
    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(gt_trans_world - trans_hat, 2, dim=-1)
    
    # Normalize it to the displacement
    normalized_rte = (rte / disp).numpy() * 1e2
    mean_normalized_rte = normalized_rte.mean()
    print("Normalized RTE: ", mean_normalized_rte)


    # <======= Evaluation on the camera pose
    if args.gt_extrinsics:
        pred_cam_pose = gt_cam
    else:
        aligned_cam_trans, cam_pose_rot = compute_pred_trans_hat(gt_cam[:,:3,3], pred_cam_pose_trans)
        pred_cam_pose_orientation = cam_pose_rot @ pred_cam_pose_orientation
        # create pred_extrinsic matrix with same shape as gt_cam
        pred_cam_pose = np.zeros_like(gt_cam)
        pred_cam_pose[:, :3, :3] = pred_cam_pose_orientation
        pred_cam_pose[:, :3, 3] = aligned_cam_trans
        pred_cam_pose[:, 3, 3] = 1


    wham["trans_world_hat"] = trans_hat
    wham["pose_world_hat"] = root_poses_hat
    wham["cam_pose_hat"] = pred_cam_pose
    wham["rte"] = mean_normalized_rte
    wham["pa_mpjpe"] = pa_mpjpe.mean()
    wham["mpjpe"] = mpjpe.mean()
    wham["w_mpjpe"] = w_mpjpe.mean()
    wham["wa_mpjpe"] = wa_mpjpe.mean()
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

    parser.add_argument("--gt_extrinsics", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, 
                        help="Use ground truth camera pose (True/False)")

    parser.add_argument("--gt_intrinsics", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, 
                        help="Use GT intrinsics (True/False)")

    parser.add_argument('--run_smplify', action='store_true', default=True,
                        help='Run Temporal SMPLify for post processing')


    parser.add_argument('-c', '--cfg', type=str, default='./configs/yamls/demo.yaml', help='cfg file path')
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Modify config options using the command-line")
    
    args = parser.parse_args()
    
    sequence_root = get_sequence_root(args, gt=True)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]

    sequence_root = get_sequence_root(args, gt=False)
    if args.gt_extrinsics:
        if args.run_smplify:
            wham_data_path = glob(os.path.join(sequence_root, "*_output_gt_camera.pkl"))[0]
            slam_path = glob(os.path.join(sequence_root, "slam_results.pth"))[0] # unused but argument is required
        else:
            wham_data_path = glob(os.path.join(sequence_root, "*_output_gt_camera_wo_SMPLify.pkl"))[0]
            slam_path = glob(os.path.join(sequence_root, "slam_results.pth"))[0]

    elif args.gt_intrinsics:
        wham_data_path = glob(os.path.join(sequence_root, "*_output_gt_intrinsics.pkl"))[0]
        slam_path = glob(os.path.join(sequence_root, "slam_results_gt_intrinsics.pth"))[0]

    else:
        wham_data_path = glob(os.path.join(sequence_root, "*_output_DPVO.pkl"))[0]
        slam_path = glob(os.path.join(sequence_root, "slam_results.pth"))[0]

    # Output folder
    if args.gt_extrinsics:
        if args.run_smplify:
            sequence = "wham_output_gt_camera_processed.pkl"
        else:
            sequence = "wham_output_gt_camera_wo_SMPLify_processed.pkl"
    elif args.gt_intrinsics:
        sequence = "wham_output_gt_intrinsics_processed.pkl"
    else:
        sequence = "wham_output_DPVO_processed.pkl"
    output_pth = osp.join(sequence_root, sequence)


    print(output_pth)
    cfg = get_cfg(args, False)
    run(gt_data_path, wham_data_path, slam_path,  output_pth, args, cfg)
