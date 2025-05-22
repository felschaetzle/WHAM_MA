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
from lib.utils.transforms import matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix
from lib.models import build_body_model

from scripts.custom_utils import get_sequence_root, find_substring
from configs import constants as _C

from scipy.spatial.transform import Rotation as R
from lib.models.smplify.custom_smplify import optimization_upper_bound, optimization_baseline, optimization_method, W_MPJPE_align, CustomSMPLify
from lib.eval.eval_utils import align_pcl

from configs.config import get_cfg_defaults
from configs.config import parse_args

from scripts.align_emdb import align_and_compute_metrics
from scripts.visualize_cam_path import invert_camera_poses, get_camera_position
from scripts.extrinsics_classifier import extrinsic_classifier
from lib.models.smplify.optimize_camera_scale import optimize_dpvo_trans
from scipy.optimize import least_squares

# torch.autograd.set_detect_anomaly(True)

def project_pts(rvec, tvec, K, pts3d):
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = (R @ pts3d.T) + tvec  # shape (3,N)
    x = pts_cam[0] / pts_cam[2]
    y = pts_cam[1] / pts_cam[2]
    uv = (K @ np.vstack([x, y, np.ones_like(x)]))[:2].T
    return uv  # shape (N,2)


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

    pose_world = torch.from_numpy(wham_raw['pose_world']).float().to(cfg.DEVICE)
    pose_world = axis_angle_to_matrix(pose_world.reshape(-1, 24, 3))

    pred['pose_body'] = pose_world[:, 1:, :, :]
    pred['poses_root_world'] = pose_world[:, :1, :, :]

    pose_root_cam = torch.from_numpy(wham_raw['poses_root_cam']).float().to(cfg.DEVICE)
    pred['poses_root_cam'] = axis_angle_to_matrix(pose_root_cam).unsqueeze(1)

    pred['cam'] = torch.from_numpy(wham_raw['cam']).float().to(cfg.DEVICE)
    pred['betas'] = torch.from_numpy(wham_raw['betas']).float().to(cfg.DEVICE)

    pred['vel_root'] = torch.from_numpy(wham_raw['vel_root_refined']).float().to(cfg.DEVICE)

    pred['poses_root_ref'] = torch.from_numpy(wham_raw['poses_root_r6d_refined']).float().to(cfg.DEVICE)

    root_r = pred['poses_root_ref'].unsqueeze(0).clone()
    root_v = pred['vel_root'].unsqueeze(0).clone()

    # from wham utils rollout_global_motion
    root = rotation_6d_to_matrix(root_r[:])
    vel_world = (root[:, :-1] @ root_v.unsqueeze(-1)).squeeze(-1)

    yup2ydown = axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(cfg.DEVICE)

    vel_world = (yup2ydown.mT @ vel_world.unsqueeze(-1)).squeeze(-1)
    pred['vel_root_world'] = vel_world
    # trans = torch.cumsum(vel_world, dim=1)
    
    bbox = torch.from_numpy(wham_raw['bbox']).float().to(cfg.DEVICE)
    res = torch.from_numpy(wham_raw['res']).float().to(cfg.DEVICE)

    kwargs = {}

    if args.load_debug:
        debug_res = joblib.load(f"output/emdb2/debug/{args.sequence}.pkl")
        for key, _ in pred.items():
            pred[key] = debug_res[key]

        extrinsics_init = debug_res["extrinsics"]
        gt_intrinsics = debug_res['intrinsics']
        input_keypoints = debug_res['input_keypoints']
        results['trans_world_align'] = debug_res['trans_world_align']
        results['pose_world_align'] = debug_res['pose_world_align'] 

        results['dpvo_extrinsics'] = debug_res['dpvo_extrinsics']
        results['wham_cam'] = debug_res['wham_cam']
        results['extrinsics_init'] = debug_res["extrinsics"].squeeze(0).clone()
 
        kp_windows = debug_res['windows']
        kp_tracks = debug_res['kp_tracks']

        pred = optimization_baseline(
            pred, input_keypoints, bbox,
            extrinsics_init, gt_intrinsics,
            smpl, cfg.DEVICE, length, res[0,:], kp_windows, kp_tracks)

    if not args.use_gt_betas:
        # Average betas
        pred['betas'] = torch.mean(pred['betas'], dim=1, keepdim=True).repeat(1, pred['betas'].shape[1], 1)

    if not args.load_debug:
        pred = W_MPJPE_align(pred, bbox, res[0], gt_intrinsics, smpl,
            cfg.DEVICE, gt_extrinsics)

        pred_root_world_aligned = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_body_pose_aligned = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)

        pred_pose_world_aligned = np.concatenate((pred_root_world_aligned, pred_body_pose_aligned), axis=-1)
        results['trans_world_align'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results['pose_world_align'] = pred_pose_world_aligned
        results['betas'] = pred['betas'].cpu().squeeze(0).numpy()

    if args.upper_bound:
        # kwargs["gt_extrinsics"] = torch.tensor(gt_extrinsics).float().to(cfg.DEVICE).unsqueeze(0)
        kwargs["gt_extrinsics"] = gt_extrinsics
        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        pred = optimization_upper_bound(
            pred, input_keypoints, bbox,
            gt_extrinsics[:,gt_data['good_frames_mask']], gt_intrinsics,
            smpl, cfg.DEVICE, length, res[0,:])
    
    if args.baseline and not args.load_debug:
        print("Get WHAM CAM")

        # Compute CAM from WHAM using intrinsics and 2d to 3d correspondence
        output = smpl.forward_align(pred['poses_body'], pred['betas'], cam_intrinsics=gt_intrinsics, 
                    bbox=bbox, res=res[:,0], trans_opt=pred['trans_world'], global_orient_opt=pred['poses_root_world'], offset=True)
        joints_3d = output.joints.reshape(*pred['cam'].shape[:2], -1, 3).squeeze(0)[:,:17,:].clone().cpu().numpy()

        intrins = gt_intrinsics.squeeze(0).cpu().numpy()

        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        
        kp_2d = input_keypoints.clone().cpu().numpy()

        kp_2d = np.ascontiguousarray(kp_2d.astype(np.float32))

        dist_coeffs = np.zeros((4, 1), dtype=np.float32) 
        WHAM_CAM = []
        rvec_prev = gt_extrinsics.clone().squeeze()[0,:3, :3].cpu().numpy()
        tvec_prev = gt_extrinsics.clone().squeeze()[0,:3, 3].cpu().numpy().reshape(3,1)
        rvec_prev = cv2.Rodrigues(rvec_prev)[0]

        conf_thresh = 0.5  # or your chosen threshold

        """        
        # smoothing weights
        λ_r = 10    # rotational velocity weight
        λ_t = 0    # translational velocity weight
        Y_r = 10    # rotational acceleration weight
        Y_t = 0    # translational acceleration weight

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        WHAM_CAM = []

        # history buffers
        rvec_prev2 = None
        tvec_prev2 = None
        rvec_prev  = None
        tvec_prev  = None

        for i in range(length):
            pts_3d = joints_3d[i]
            pts_2d_full = kp_2d[i]
            conf = pts_2d_full[:, 2]
            mask = conf > conf_thresh
            if mask.sum() < 6:
                print(f"[Frame {i}] Not enough keypoints—skipping.")
                continue

            obj_pts = pts_3d[mask].astype(np.float32)
            img_pts = pts_2d_full[mask, :2].astype(np.float32)
            K = intrins.astype(np.float64)

            # initial guess
            if rvec_prev is None:
                _, rvec_prev, tvec_prev = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs,
                                                    flags=cv2.SOLVEPNP_EPNP)
            x0 = np.concatenate([rvec_prev.ravel(), tvec_prev.ravel()])

            def residuals(x):
                # unpack
                r = x[0:3].reshape(3,1)
                t = x[3:6].reshape(3,1)

                # reprojection residuals
                proj = project_pts(r, t, K, obj_pts)
                reproj_err = (proj - img_pts).ravel()

                # velocity (first‐order) smoothness
                if rvec_prev is None:
                    vel_r_err = np.zeros(3)
                    vel_t_err = np.zeros(3)
                else:
                    vel_r_err = x[0:3] - rvec_prev.ravel()
                    vel_t_err = x[3:6] - tvec_prev.ravel()
                smooth_err = np.hstack([
                    np.sqrt(λ_r) * vel_r_err,
                    np.sqrt(λ_t) * vel_t_err
                ])

                # acceleration (second‐order) smoothness
                if rvec_prev2 is None:
                    accel_err = np.zeros(6)
                else:
                    accel_r = x[0:3] \
                            - 2 * rvec_prev.ravel() \
                            + rvec_prev2.ravel()
                    accel_t = x[3:6] \
                            - 2 * tvec_prev.ravel() \
                            + tvec_prev2.ravel()
                    accel_err = np.hstack([
                        np.sqrt(Y_r) * accel_r,
                        np.sqrt(Y_t) * accel_t
                    ])

                return np.concatenate([reproj_err, smooth_err, accel_err])

            res = least_squares(residuals, x0, method='lm',
                                max_nfev=50, xtol=1e-6, ftol=1e-6)

            # extract optimized pose
            r_opt = res.x[0:3].reshape(3,1)
            t_opt = res.x[3:6].reshape(3,1)

            # shift history
            rvec_prev2, rvec_prev = rvec_prev, r_opt.copy()
            tvec_prev2, tvec_prev = tvec_prev, t_opt.copy()

            # assemble 4×4
            Rm, _ = cv2.Rodrigues(r_opt)
            T = np.eye(4, dtype=np.float32)
            T[:3,:3] = Rm
            T[:3, 3] = t_opt.squeeze()
            WHAM_CAM.append(T)
            """
        
        for i in range(length):
            pts_3d = joints_3d[i]
            pts_2d_full = kp_2d[i]

            conf = pts_2d_full[:, 2]
            valid_mask = conf > conf_thresh

            if valid_mask.sum() < 6:
                print(f"[Frame {i}] Not enough valid keypoints ({valid_mask.sum()}) — skipping.")
                continue

            pts_3d_valid = np.ascontiguousarray(pts_3d[valid_mask].astype(np.float32))
            pts_2d_valid = np.ascontiguousarray(pts_2d_full[valid_mask][:, :2].astype(np.float32))
            K = np.ascontiguousarray(intrins.astype(np.float32))

            # Use previous frame's result as initial guess, if available
            if rvec_prev is not None and tvec_prev is not None:
                success, rvec, tvec = cv2.solvePnP(
                    pts_3d_valid, pts_2d_valid, K, dist_coeffs,
                    rvec=rvec_prev, tvec=tvec_prev,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                print("SHIIIT")

            if not success:
                print(f"[Frame {i}] solvePnP failed.")
                continue

            rvec_prev = rvec.copy()
            tvec_prev = tvec.copy()

            R_new, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R_new
            T[:3, 3] = tvec.squeeze()
            WHAM_CAM.append(T)

        WHAM_CAM = np.array(WHAM_CAM)
        wham_extrinsics = torch.from_numpy(WHAM_CAM).float().to(cfg.DEVICE).unsqueeze(0)
        pred['wham_cam'] = wham_extrinsics
        results['wham_cam_raw'] = wham_extrinsics.clone().squeeze(0).cpu().numpy()

        custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=10, res=res, device=cfg.DEVICE)
        pred = custom_smplify.smooth_extrinsics(pred, input_keypoints, bbox, wham_extrinsics, gt_intrinsics)
        wham_extrinsics = pred['wham_cam'].clone()
        results['wham_cam'] = wham_extrinsics.squeeze(0).cpu().numpy()

        dpvo_path = _C.PATHS.WHAM_OUTPUT + "/" + args.subject + "_" + args.sequence + "/slam_results_gt_intrinsics.pth"
        dpvo_output = joblib.load(dpvo_path)

        dpvo_orientation = R.from_quat(dpvo_output[:,3:]).as_matrix()
        dpvo_trans = dpvo_output[:,:3]

        # Create 4x4 transformation matrices for dpvo_cam
        dpvo_cam = np.eye(4)[None].repeat(len(dpvo_orientation), axis=0)
        dpvo_cam[:, :3, :3] = dpvo_orientation
        dpvo_cam[:, :3, 3] = dpvo_trans
        dpvo_extrinsics = invert_camera_poses(dpvo_cam)
        dpvo_extrinsics = torch.from_numpy(dpvo_extrinsics).float().to(cfg.DEVICE)

        results['dpvo_extrinsics_unscaled'] = dpvo_extrinsics.clone()

        aux_dpvo = dpvo_extrinsics @ gt_extrinsics[0,0]
        aux_dpvo_cam_pose = get_camera_position(aux_dpvo.cpu())
        aux_wham_cam_pose = get_camera_position(pred['wham_cam'].squeeze().cpu())
        
        scale, _, _ = align_pcl(aux_wham_cam_pose.cpu(), aux_dpvo_cam_pose[gt_data['good_frames_mask']])
        
        print("DPVO scale: ", scale)
        dpvo_extrinsics[:, :3, 3] *= float(scale)
        dpvo_extrinsics = dpvo_extrinsics @ gt_extrinsics[0,0]

        results['dpvo_extrinsics'] = dpvo_extrinsics.clone().cpu().numpy()
        # results['dpvo_scale'] = scale
        dpvo_extrinsics = dpvo_extrinsics[gt_data['good_frames_mask']].unsqueeze(0)
        
        results['extrinsics_init'] = dpvo_extrinsics.squeeze(0).cpu().numpy()

        if args.save_debug:
            debug_res = {}
            for key, value in pred.items():
                debug_res[key] = value
            debug_res["extrinsics"] = extrinsics_init
            debug_res['intrinsics'] = gt_intrinsics
            debug_res['input_keypoints'] = input_keypoints

            debug_res['trans_world_align'] = results['trans_world_align']
            debug_res['pose_world_align'] = results['pose_world_align']

            debug_res['dpvo_extrinsics'] = results['dpvo_extrinsics']
            debug_res['wham_cam'] = results['wham_cam']
            debug_res['windows'] = kp_windows
            debug_res['kp_tracks'] = kp_tracks

            joblib.dump(debug_res, f"output/emdb2/debug/{args.sequence}.pkl")
            print("save debug pkl to", f"output/emdb2/debug/{args.sequence}.pkl")
            return

        pred = optimization_baseline(
            pred, input_keypoints, bbox,
            dpvo_extrinsics, gt_intrinsics,
            smpl, cfg.DEVICE, length, res[0,:])
    
    if args.method and not args.load_debug:
        print("Get WHAM CAM")

        # Compute CAM from WHAM using intrinsics and 2d to 3d correspondence
        output = smpl.forward_align(pred['poses_body'], pred['betas'], cam_intrinsics=gt_intrinsics, 
                    bbox=bbox, res=res[:,0], trans_opt=pred['trans_world'], global_orient_opt=pred['poses_root_world'], offset=True)
        joints_3d = output.joints.reshape(*pred['cam'].shape[:2], -1, 3).squeeze(0)[:,:17,:].clone().cpu().numpy()

        intrins = gt_intrinsics.squeeze(0).cpu().numpy()

        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        
        kp_2d = input_keypoints.clone().cpu().numpy()

        kp_2d = np.ascontiguousarray(kp_2d.astype(np.float32))

        dist_coeffs = np.zeros((4, 1), dtype=np.float32) 
        WHAM_CAM = []
        rvec_prev = gt_extrinsics.clone().squeeze()[0,:3, :3].cpu().numpy()
        tvec_prev = gt_extrinsics.clone().squeeze()[0,:3, 3].cpu().numpy().reshape(3,1)
        rvec_prev = cv2.Rodrigues(rvec_prev)[0]

        conf_thresh = 0.5  # or your chosen threshold

        """        
        # smoothing weights
        λ_r = 10    # rotational velocity weight
        λ_t = 0    # translational velocity weight
        Y_r = 10    # rotational acceleration weight
        Y_t = 0    # translational acceleration weight

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        WHAM_CAM = []

        # history buffers
        rvec_prev2 = None
        tvec_prev2 = None
        rvec_prev  = None
        tvec_prev  = None

        for i in range(length):
            pts_3d = joints_3d[i]
            pts_2d_full = kp_2d[i]
            conf = pts_2d_full[:, 2]
            mask = conf > conf_thresh
            if mask.sum() < 6:
                print(f"[Frame {i}] Not enough keypoints—skipping.")
                continue

            obj_pts = pts_3d[mask].astype(np.float32)
            img_pts = pts_2d_full[mask, :2].astype(np.float32)
            K = intrins.astype(np.float64)

            # initial guess
            if rvec_prev is None:
                _, rvec_prev, tvec_prev = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs,
                                                    flags=cv2.SOLVEPNP_EPNP)
            x0 = np.concatenate([rvec_prev.ravel(), tvec_prev.ravel()])

            def residuals(x):
                # unpack
                r = x[0:3].reshape(3,1)
                t = x[3:6].reshape(3,1)

                # reprojection residuals
                proj = project_pts(r, t, K, obj_pts)
                reproj_err = (proj - img_pts).ravel()

                # velocity (first-order) smoothness
                if rvec_prev is None:
                    vel_r_err = np.zeros(3)
                    vel_t_err = np.zeros(3)
                else:
                    vel_r_err = x[0:3] - rvec_prev.ravel()
                    vel_t_err = x[3:6] - tvec_prev.ravel()
                smooth_err = np.hstack([
                    np.sqrt(λ_r) * vel_r_err,
                    np.sqrt(λ_t) * vel_t_err
                ])

                # acceleration (second-order) smoothness
                if rvec_prev2 is None:
                    accel_err = np.zeros(6)
                else:
                    accel_r = x[0:3] \
                            - 2 * rvec_prev.ravel() \
                            + rvec_prev2.ravel()
                    accel_t = x[3:6] \
                            - 2 * tvec_prev.ravel() \
                            + tvec_prev2.ravel()
                    accel_err = np.hstack([
                        np.sqrt(Y_r) * accel_r,
                        np.sqrt(Y_t) * accel_t
                    ])

                return np.concatenate([reproj_err, smooth_err, accel_err])

            res = least_squares(residuals, x0, method='lm',
                                max_nfev=50, xtol=1e-6, ftol=1e-6)

            # extract optimized pose
            r_opt = res.x[0:3].reshape(3,1)
            t_opt = res.x[3:6].reshape(3,1)

            # shift history
            rvec_prev2, rvec_prev = rvec_prev, r_opt.copy()
            tvec_prev2, tvec_prev = tvec_prev, t_opt.copy()

            # assemble 4x4
            Rm, _ = cv2.Rodrigues(r_opt)
            T = np.eye(4, dtype=np.float32)
            T[:3,:3] = Rm
            T[:3, 3] = t_opt.squeeze()
            WHAM_CAM.append(T)
            """
        
        for i in range(length):
            pts_3d = joints_3d[i]
            pts_2d_full = kp_2d[i]

            conf = pts_2d_full[:, 2]
            valid_mask = conf > conf_thresh

            if valid_mask.sum() < 6:
                print(f"[Frame {i}] Not enough valid keypoints ({valid_mask.sum()}) — skipping.")
                continue

            pts_3d_valid = np.ascontiguousarray(pts_3d[valid_mask].astype(np.float32))
            pts_2d_valid = np.ascontiguousarray(pts_2d_full[valid_mask][:, :2].astype(np.float32))
            K = np.ascontiguousarray(intrins.astype(np.float32))

            # Use previous frame's result as initial guess, if available
            if rvec_prev is not None and tvec_prev is not None:
                success, rvec, tvec = cv2.solvePnP(
                    pts_3d_valid, pts_2d_valid, K, dist_coeffs,
                    rvec=rvec_prev, tvec=tvec_prev,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                print("SHIIIT")

            if not success:
                print(f"[Frame {i}] solvePnP failed.")
                continue

            rvec_prev = rvec.copy()
            tvec_prev = tvec.copy()

            R_new, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R_new
            T[:3, 3] = tvec.squeeze()
            WHAM_CAM.append(T)

        WHAM_CAM = np.array(WHAM_CAM)
        wham_extrinsics = torch.from_numpy(WHAM_CAM).float().to(cfg.DEVICE).unsqueeze(0)
        pred['wham_cam'] = wham_extrinsics
        results['wham_cam_raw'] = wham_extrinsics.clone().squeeze(0).cpu().numpy()

        custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=10, res=res, device=cfg.DEVICE)
        pred = custom_smplify.smooth_extrinsics(pred, input_keypoints, bbox, wham_extrinsics, gt_intrinsics)
        wham_extrinsics = pred['wham_cam'].clone()
        results['wham_cam'] = wham_extrinsics.squeeze(0).cpu().numpy()

        dpvo_path = _C.PATHS.WHAM_OUTPUT + "/" + args.subject + "_" + args.sequence + "/slam_results_gt_intrinsics.pth"
        dpvo_output = joblib.load(dpvo_path)

        dpvo_orientation = R.from_quat(dpvo_output[:,3:]).as_matrix()
        dpvo_trans = dpvo_output[:,:3]

        # Create 4x4 transformation matrices for dpvo_cam
        dpvo_cam = np.eye(4)[None].repeat(len(dpvo_orientation), axis=0)
        dpvo_cam[:, :3, :3] = dpvo_orientation
        dpvo_cam[:, :3, 3] = dpvo_trans
        dpvo_extrinsics = invert_camera_poses(dpvo_cam)
        dpvo_extrinsics = torch.from_numpy(dpvo_extrinsics).float().to(cfg.DEVICE)

        results['dpvo_extrinsics_unscaled'] = dpvo_extrinsics.clone()

        aux_dpvo = dpvo_extrinsics @ gt_extrinsics[0,0]
        aux_dpvo_cam_pose = get_camera_position(aux_dpvo.cpu())
        aux_wham_cam_pose = get_camera_position(pred['wham_cam'].squeeze().cpu())

        # scales = []
        # for i in range(length):
        #     if i> 0 and i % 100 == 0:
        #         scale, _, _ = align_pcl(aux_wham_cam_pose.cpu()[i-100:i], aux_dpvo_cam_pose[gt_data['good_frames_mask']][i-100:i])
        #         scales.append(scale)
        # scale = torch.mean(torch.tensor(scales))
        
        scale, _, _ = align_pcl(aux_wham_cam_pose.cpu(), aux_dpvo_cam_pose[gt_data['good_frames_mask']])
        
        print("DPVO scale: ", scale)
        dpvo_extrinsics[:, :3, 3] *= float(scale)
        dpvo_extrinsics = dpvo_extrinsics @ gt_extrinsics[0,0]

        results['dpvo_extrinsics'] = dpvo_extrinsics.clone().cpu().numpy()
        # results['dpvo_scale'] = scale
        dpvo_extrinsics = dpvo_extrinsics[gt_data['good_frames_mask']].unsqueeze(0)
        
        # classify the correct camera for each frame
        extrinsics_init, kp_windows, kp0, kp1, _ = extrinsic_classifier(args, wham_extrinsics.squeeze(0).cpu().numpy(), dpvo_extrinsics.squeeze(0).cpu().numpy())
        extrinsics_init = torch.from_numpy(extrinsics_init).float().to(cfg.DEVICE).unsqueeze(0)
        kp_tracks = (kp0, kp1)
        results['extrinsics_init'] = extrinsics_init.clone().squeeze(0).cpu().numpy()

        # pth = osp.join(output_pth, "baseline_gt_betas.pkl")
        # joblib.dump(results, pth)
        # return
        if args.save_debug:
            debug_res = {}
            for key, value in pred.items():
                debug_res[key] = value
            debug_res["extrinsics"] = extrinsics_init
            debug_res['intrinsics'] = gt_intrinsics
            debug_res['input_keypoints'] = input_keypoints

            debug_res['trans_world_align'] = results['trans_world_align']
            debug_res['pose_world_align'] = results['pose_world_align']

            debug_res['dpvo_extrinsics'] = results['dpvo_extrinsics']
            debug_res['wham_cam'] = results['wham_cam']
            debug_res['windows'] = kp_windows
            debug_res['kp_tracks'] = kp_tracks

            joblib.dump(debug_res, f"output/emdb2/debug/{args.sequence}.pkl")
            print("save debug pkl to", f"output/emdb2/debug/{args.sequence}.pkl")
            return

        pred = optimization_method(
            pred, input_keypoints, bbox,
            extrinsics_init, gt_intrinsics,
            smpl, cfg.DEVICE, length, res[0,:], kp_windows, kp_tracks)

    # ========= Store results ========= #
    pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)

    pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)

    pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
    pred_pose_cam = matrix_to_axis_angle(pred['poses_root_cam'].squeeze(1)).cpu().numpy().reshape(-1, 3)

    pred_root_world_init = matrix_to_axis_angle(pred['poses_root_world_init']).cpu().numpy().reshape(-1, 3)
    results['poses_root_world_init'] = pred_root_world_init

    results['pose_world'] = pred_pose_world
    results['poses_root_cam'] = pred_pose_cam
    results['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
    results['trans_world_init'] = pred['trans_world_init'].cpu().squeeze(0).numpy()
    results['betas'] = pred['betas'].cpu().squeeze(0).numpy()
    results['bbox'] = bbox.cpu().numpy()
    results['cam'] = pred['cam'].cpu().numpy()
    results['res'] = res[0].cpu().numpy()

    if args.method:
        results['optimized_cam'] = pred['optimized_cam'].cpu().numpy()
        results['num_windows_used_epipolar'] = pred['num_windows_used_epipolar']
    else:
        results['optimized_cam'] = dpvo_extrinsics.squeeze(0).cpu().numpy()
    # results['optimized_cam_recovered'] = pred['optimized_cam_t'].cpu().numpy()


    
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
    elif args.method:
        pth = osp.join(output_pth, "method.pkl")
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