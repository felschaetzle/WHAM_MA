import pickle
import joblib
import numpy as np
import torch
from smplx import SMPL
import sys
sys.path.append('/home/felix/WHAM_MA')
from lib.models import build_network, build_body_model
from configs.config import get_cfg

import argparse
from glob import glob
import os
import os.path as osp


from scripts.custom_utils import open_pkl

from lib.utils.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from lib.eval.eval_utils import first_align_joints_return_R_t, compute_pred_trans_hat, global_align_joints, compute_rte, first_align_joints, align_pcl, compute_jpe, batch_align_by_pelvis, batch_compute_similarity_transform_torch
from scripts.custom_utils import get_sequence_root

import sys
sys.path.append("/home/felix/WHAM_MA")
from configs import constants as _C

from scipy.spatial.transform import Rotation as R
from lib.utils import transforms
from configs.config import parse_args
from lib.models.smpl import convert_pare_to_full_img_cam

m2mm = 1e3
pelvis_idxs = [1, 2]

def align_and_compute_metrics(gt_pth, wham_pth, cfg):

    yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float()

    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE) 

    # smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    # smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
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
    
    gt_trans_world = torch.from_numpy(gt_trans_world)
    pred_trans_world = torch.from_numpy(pred_trans_world)

    gt_pose_world = torch.from_numpy(gt_pose_world).unsqueeze(0).float()
    pred_pose_world = torch.from_numpy(pred_pose_world).unsqueeze(0).float()
   
    # <======= Evaluation on the local motion
    pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = batch_align_by_pelvis(
        [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam], pelvis_idxs
    )
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d_cam, target_j3d_cam)
    pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy() * m2mm
    mpjpe = torch.sqrt(((pred_j3d_cam - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).detach().cpu().numpy() * m2mm
    # print("PA-MPJPE: ", pa_mpjpe.mean())
    # print("MPJPE: ", mpjpe.mean())

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

    # print("W-MPJPE: ", w_mpjpe.mean())
    print("WA-MPJPE: ", wa_mpjpe.mean())

    # trans_hat, rot = compute_pred_trans_hat(gt_trans_world, pred_trans_world)

    # # align joint from wham[0] to cam[0]
    # wham_joints_cam, R_wham_cam, t_wham_cam = first_align_joints_return_R_t(align_pred_j3d_cam.joints.cpu(), align_pred_j3d_wham.joints.cpu())
    # R_wham_cam = R_wham_cam.to(cfg.DEVICE)
    # t_wham_cam = t_wham_cam.to(cfg.DEVICE)
    
    # initial_extrinsics = gt_cam[0]
    # cam_pose = np.linalg.inv(initial_extrinsics)
    # R_cam_pose = torch.tensor(cam_pose[:3, :3]).unsqueeze(0).float().to(cfg.DEVICE)
    # t_cam_pose = torch.tensor(cam_pose[:3, 3]).unsqueeze(0).float().to(cfg.DEVICE)

    # # apply to translation
    # transl_cam = (R_wham_cam @ pred_trans_world.to(cfg.DEVICE).unsqueeze(-1)).squeeze(-1) + t_wham_cam
    # trans_hat = (R_cam_pose @ transl_cam.unsqueeze(-1)).squeeze(-1) + t_cam_pose
    # # apply to rotation
    # root_poses_hat = R_cam_pose @ R_wham_cam @ pred_pose_world.to(cfg.DEVICE)


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
    rte = torch.norm(gt_trans_world - trans_hat.cpu(), 2, dim=-1)
    
    # Normalize it to the displacement
    rte = compute_rte(gt_trans_world, pred_trans_world) * 1e2
    mean_rte = rte.mean()
    print("Normalized RTE: ", mean_rte)


    wham["trans_world_hat"] = trans_hat.cpu().detach().numpy()
    wham["pose_world_hat"] = root_poses_hat
    wham["rte"] = mean_rte
    wham["pa_mpjpe"] = pa_mpjpe.mean()
    wham["mpjpe"] = mpjpe.mean()
    wham["w_mpjpe"] = w_mpjpe.mean()
    wham["wa_mpjpe"] = wa_mpjpe.mean()
    joblib.dump(wham, wham_pth)
    print("Results saved to: ", wham_pth)

    print("DONE")


if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)


    sequence_root = get_sequence_root(args, gt=True)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]

    sequence_root = get_sequence_root(args, gt=False)
    if args.run_smplify:
        if args.naive_intrinsics:
            wham_data_path = glob(os.path.join(sequence_root, "smplify_naive_intrinsics.pkl"))[0]
        else:
            wham_data_path = glob(os.path.join(sequence_root, "smplify.pkl"))[0]
    elif args.baseline:
        wham_data_path = glob(os.path.join(sequence_root, "baseline.pkl"))[0]
    else:
        wham_data_path = glob(os.path.join(sequence_root, "eval.pkl"))[0]

    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl_align = build_body_model(cfg.DEVICE, smpl_batch_size)

    print("Align: ", wham_data_path)
    align_and_compute_metrics(gt_data_path, wham_data_path, cfg)
