import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from lib.data.dataloader import setup_eval_dataloader
from lib.utils.transforms import matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d
from lib.models import build_body_model

from scripts.custom_utils import get_sequence_root, find_substring
from configs import constants as _C

from scipy.spatial.transform import Rotation as R
from lib.models.smplify.custom_smplify import optimization_upper_bound, optimization_baseline, W_MPJPE_align
from lib.eval.eval_utils import align_pcl

from configs.config import get_cfg_defaults
from configs.config import parse_args

from scripts.align_emdb import align_and_compute_metrics
from scripts.visualize_cam_path import invert_camera_poses, get_camera_position


def run(cfg,
        args,
        video,
        output_pth):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    calib = _C.PATHS.WHAM_OUTPUT + "/" + args.subject + "_" + args.sequence + "/gt_intrinsics.txt"

    eval_loader = setup_eval_dataloader(cfg, 'emdb', args.eval_split, cfg.MODEL.BACKBONE)
    emdb_sequence_index = find_substring(args.subject+"_"+args.sequence, eval_loader.dataset.labels['vid'])
    if emdb_sequence_index is None:
        logger.error(f"Sequence {args.subject}_{args.sequence} not found in the emdb2 dataset. Not usefull for global trajectory.")
        return

    print("Loading data from eval loader")

    length_update = eval_loader.dataset.labels['frame_id'][emdb_sequence_index].shape[0] - 1
    print("Found # frames in dataset: ", length)
    print("Found # frames in eval loader: ", length_update)
    length = length_update

    calib_data = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib_data[:4]
    gt_intrinsics = np.eye(3)
    gt_intrinsics[0,0] = fx
    gt_intrinsics[0,2] = cx
    gt_intrinsics[1,1] = fy
    gt_intrinsics[1,2] = cy
    gt_intrinsics = torch.tensor(gt_intrinsics).float().to(cfg.DEVICE).unsqueeze(0)

    sequence_root = get_sequence_root(args)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    gt_data = joblib.load(gt_data_path)
    gt_extrinsics = gt_data["camera"]["extrinsics"]
    gt_cam_pose = np.linalg.inv(gt_extrinsics)
    gt_cam_pose_rot = R.from_matrix(gt_cam_pose[:,:3,:3]).as_quat()
    gt_cam_pose = np.concatenate([gt_cam_pose[:,:3,3], gt_cam_pose_rot], axis=1)

    results = defaultdict(dict)
    gt_extrinsics = torch.tensor(gt_extrinsics).float().to(cfg.DEVICE).unsqueeze(0)



    if args.use_gt_betas:
        wham_raw = joblib.load(_C.PATHS.WHAM_OUTPUT + "/" + args.subject + "_" + args.sequence + "/wham_raw_output_gt_betas.pkl")
    else:
        wham_raw = joblib.load(_C.PATHS.WHAM_OUTPUT + "/" + args.subject + "_" + args.sequence + "/wham_raw_output.pkl")
    pred = {}
    
    pred['trans_world'] = torch.from_numpy(wham_raw['trans_world']).float().to(cfg.DEVICE)

    pred['poses_body'] = torch.from_numpy(wham_raw['poses_body']).float().to(cfg.DEVICE)
    pred['poses_root_world'] = torch.from_numpy(wham_raw['poses_root_world']).float().to(cfg.DEVICE)
    pred['poses_root_cam'] = torch.from_numpy(wham_raw['poses_root_cam']).float().to(cfg.DEVICE)

    pred['cam'] = torch.from_numpy(wham_raw['cam']).float().to(cfg.DEVICE)
    pred['betas'] = torch.from_numpy(wham_raw['betas']).float().to(cfg.DEVICE)

    bbox = torch.from_numpy(wham_raw['bbox']).float().to(cfg.DEVICE)
    res = torch.from_numpy(wham_raw['res']).float().to(cfg.DEVICE)

    kwargs = {}
    kwargs['bbox'] = bbox
    kwargs['res'] = res


    if not args.use_gt_betas:
        # Average betas
        pred['betas'] = torch.mean(pred['betas'], dim=1, keepdim=True).repeat(1, pred['betas'].shape[1], 1)

    pred = W_MPJPE_align(pred, kwargs['bbox'], kwargs['res'][0], gt_intrinsics, smpl,
                cfg.DEVICE, gt_extrinsics)

    pred_root_world_aligned = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
    pred_body_pose_aligned = matrix_to_axis_angle(rotation_6d_to_matrix(pred['poses_body'].reshape(-1,23,6))).cpu().numpy().reshape(-1, 69)

    pred_pose_world_aligned = np.concatenate((pred_root_world_aligned, pred_body_pose_aligned), axis=-1)
    results['trans_world_align'] = pred['trans_world'].cpu().squeeze(0).numpy()
    results['pose_world_align'] = pred_pose_world_aligned

    if args.upper_bound:
        # kwargs["gt_extrinsics"] = torch.tensor(gt_extrinsics).float().to(cfg.DEVICE).unsqueeze(0)
        kwargs["gt_extrinsics"] = gt_extrinsics
        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        pred = optimization_upper_bound(
            pred, input_keypoints, kwargs['bbox'],
            kwargs['gt_extrinsics'], gt_intrinsics,
            smpl, cfg.DEVICE, length, kwargs['res'][0,:])
    
    if args.baseline:
        dpvo_path = _C.PATHS.WHAM_OUTPUT + "/" + args.subject + "_" + args.sequence + "/slam_results_gt_intrinsics.pth"
        dpvo_output = joblib.load(dpvo_path)
        print(len(dpvo_output))
        dpvo_orientation = R.from_quat(dpvo_output[:,3:]).as_matrix()
        dpvo_trans = dpvo_output[:,:3]
        # create dpvo_cam object

        # Create 4x4 transformation matrices for dpvo_cam
        dpvo_cam = np.eye(4)[None].repeat(len(dpvo_orientation), axis=0)
        dpvo_cam[:, :3, :3] = dpvo_orientation
        dpvo_cam[:, :3, 3] = dpvo_trans
        dpvo_extrinsics = invert_camera_poses(dpvo_cam)
        dpvo_extrinsics = torch.from_numpy(dpvo_extrinsics).float().to(cfg.DEVICE)


        #estimate scale
        # scale = 14.489 # P8 90
        results['dpvo_extrinsics_unscaled'] = dpvo_extrinsics.clone()
        aux_dpvo = dpvo_extrinsics @ gt_extrinsics[0,0]
        aux_dpvo_cam_pose = get_camera_position(aux_dpvo.cpu())


        scale, _, _ = align_pcl(pred['trans_world'].unsqueeze(0).cpu(), aux_dpvo_cam_pose[gt_data['good_frames_mask']].unsqueeze(0))
        print(scale)
        dpvo_extrinsics[:, :3, 3] *= float(scale)
        dpvo_extrinsics = dpvo_extrinsics @ gt_extrinsics[0,0]

        results['dpvo_extrinsics'] = dpvo_extrinsics.clone().cpu().numpy()

        kwargs["gt_extrinsics"] = dpvo_extrinsics.unsqueeze(0)
        
        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        
        pred = optimization_baseline(
            pred, input_keypoints, kwargs['bbox'],
            kwargs['gt_extrinsics'], gt_intrinsics,
            smpl, cfg.DEVICE, length, kwargs['res'][0,:])
        
        # results['dpvo_extrinsics'] = dpvo_extrinsics.cpu().numpy()
        results['dpvo_scale'] = scale
        # results['trans_world_align'] = pred_align['trans_world'].cpu().squeeze(0).numpy()
        # pred_root_world_aligned = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        # results['pose_world_align'] = pred_root_world_aligned
    # ========= Store results ========= #
    pred_body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(pred['poses_body'].reshape(-1,23,6))).cpu().numpy().reshape(-1, 69)

    pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)

    pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
    pred_pose_cam = matrix_to_axis_angle(pred['poses_root_cam'].squeeze(1)).cpu().numpy().reshape(-1, 3)

    results['pose_world'] = pred_pose_world
    results['poses_root_cam'] = pred_pose_cam
    results['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
    results['betas'] = pred['betas'].cpu().squeeze(0).numpy()
    results['bbox'] = kwargs['bbox'].cpu().numpy()
    results['cam'] = pred['cam'].cpu().numpy()
    results['res'] = kwargs['res'][0].cpu().numpy()
    

    if args.upper_bound:
        if args.use_gt_betas:
            pth = osp.join(output_pth, "upper_bound_gt_betas.pkl")
        else:
            pth = osp.join(output_pth, "upper_bound.pkl")
        joblib.dump(results, pth)
        print("Save results to ", pth)
    elif args.baseline:
        if args.use_gt_betas:
            pth = osp.join(output_pth, "baseline_gt_betas.pkl")
        else:
            pth = osp.join(output_pth, "baseline.pkl")
        joblib.dump(results, pth)
        print("Save results to ", pth)
    else:
        pth = osp.join(output_pth, "eval.pkl")
        joblib.dump(results, pth)
        print("Save results to ", pth)

    align_and_compute_metrics(gt_data_path, pth, cfg)

if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    
    sequence_root = get_sequence_root(args)

    # Output folder
    sequence = args.subject + "_" + args.sequence
    video_path = glob(os.path.join(sequence_root, "*.mov"))[0]
    print(sequence)
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg,
        args, 
        video_path,
        output_pth)
        
    logger.info('Done !')