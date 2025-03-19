import os
import torch
from tqdm import tqdm
import numpy as np

from lib.models import build_body_model
from .custom_losses import CustomSMPLifyLoss, create_SMPL_param_closure
from lib.models.smpl import convert_pare_to_full_img_cam
from lib.utils.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
from lib.eval.eval_utils import first_align_joints_return_R_t
from matplotlib import pyplot as plt
import cv2
from lib.models.smpl import full_perspective_projection

import joblib
class CustomSMPLify():
    
    def __init__(self, 
                 smpl=None,
                 lr=1e-2,
                 num_iters=5,
                 num_steps=10,
                 img_w=None,
                 img_h=None,
                 device=None,
                 ):
        
        self.smpl = smpl
        self.lr = lr
        self.num_iters = num_iters
        self.num_steps = num_steps
        self.img_w = img_w
        self.img_h = img_h
        self.device = device

    def fit(self, init_pred, keypoints, bbox, gt_extrinsics, cam_intrinsics, joints3d_world):
        
        def to_params(param):
            return param.requires_grad_(True)
        
        pose = init_pred['pose']
        betas = init_pred['betas']
        cam = init_pred['cam']

        transl_world = init_pred['trans_world'].squeeze(0)
        poses_root_world = init_pred['poses_root_world'].squeeze(0)
        
        # Stage 1. Optimize translation
        params = [to_params(pose), to_params(betas), to_params(cam), to_params(transl_world), to_params(poses_root_world), to_params(joints3d_world)]
        optim_params = [params[5]]
        
        optimizer = torch.optim.LBFGS(
            optim_params, 
            lr=self.lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        loss_fn = CustomSMPLifyLoss(cam_intrinsics, init_pose=pose, device=self.device, gt_extrinsics=gt_extrinsics)
        
        closure = loss_fn.create_closure(optimizer,
                    self.smpl, 
                    params,
                    bbox,
                    keypoints
                    )
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            # print(j, msg)
            j_bar.set_postfix_str(msg)

        print(f"Final joint opt loss: {loss.item():.1f}")

        init_pred['pose'] = params[0].detach()
        init_pred['betas'] = params[1].detach()
        init_pred['cam'] = params[2].detach()
        init_pred['trans_world'] = params[3].detach()
        init_pred['poses_root_world'] = params[4].detach()
        init_pred['joints3d_world'] = params[5].detach()
        
        return init_pred
    
    def fit_SMPL_params(self, init_pred, transl_world_aligned, poses_root_world_aligned):
        
        def to_params(param):
            return param.requires_grad_(True)
        
        pose = init_pred['pose']
        betas = init_pred['betas']
        cam = init_pred['cam']

        transl_world = transl_world_aligned
        poses_root_world = poses_root_world_aligned
        joints3d_world = init_pred['joints3d_world']

        lr = self.lr
        
        # Stage 1. Optimize translation
        params = [to_params(pose), to_params(betas), to_params(cam), to_params(transl_world), to_params(poses_root_world), to_params(joints3d_world)]

        # SMPL param recovery
        optim_params = [params[3], params[4]]
        optimizer_smpl_params = torch.optim.LBFGS(
            optim_params, 
            lr=lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
                
        closure_smpl_params = create_SMPL_param_closure(optimizer_smpl_params,
                    self.smpl, 
                    params
                    )
        
        for j in (j_bar := tqdm(range(5), leave=False)):
            optimizer_smpl_params.zero_grad()
            loss = optimizer_smpl_params.step(closure_smpl_params)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)

        print(f"Final SMPL param opt loss: {loss.item():.1f}")

        init_pred['pose'] = params[0].detach()
        init_pred['betas'] = params[1].detach()
        init_pred['cam'] = params[2].detach()
        init_pred['trans_world'] = params[3].detach()
        init_pred['poses_root_world'] = params[4].detach()
        init_pred['joints3d_world'] = params[5].detach()
        
        return init_pred

# =============================================================================
# Progressive Optimization Function
# =============================================================================
def progressive_global_translation_optimization(init_pred, keypoints, bbox,
                                                  gt_extrinsics, cam_intrinsics,
                                                  smpl, device,
                                                  length, res):
    """
    Optimize global translation progressively over increasing frame windows.
    
    Args:
        init_pred: Dictionary of initial SMPL parameters (e.g., 'pose', 'betas', 'cam',
                   'trans_world', 'poses_root_world') with shape [T, ...].
        keypoints: Tensor of 2D keypoints [T, num_keypoints, 3] (last channel = confidence).
        bbox: Tensor of bounding boxes per frame.
        gt_extrinsics: Ground-truth extrinsics tensor [T, 4, 4] (or [T, 1, 4, 4]).
        cam_intrinsics: Camera intrinsics tensor [T, 3, 3] (or [T, 1, 3, 3]).
        smpl: Your SMPL model.
        device: Torch device.
        window_steps: List of frame counts for progressive optimization.
        img_w, img_h: Image width and height.
        
    Returns:
        current_pred: Dictionary with updated (optimized) parameters.
        optimized_results: Dictionary mapping window size to the optimized parameters.
    """

    # Create an instance of CustomSMPLify
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=10,
                                   img_w=res[0], img_h=res[1], device=device)
    
    pose = init_pred['pose']
    betas = init_pred['betas']
    cam = init_pred['cam']

    transl_wham_world = init_pred['trans_world'].squeeze(0)
    poses_root_wham_world = init_pred['poses_root_world'].squeeze(0).unsqueeze(1)
    window_size = 100
    # get transl and root_pose in gt world frame
    joints3d_world, transl_world_aligned, poses_root_world_aligned = W_MPJPE_align(cam, bbox, res, cam_intrinsics, smpl, device, pose, betas, transl_wham_world, poses_root_wham_world, gt_extrinsics, window_size)

    # Copy the initial predictions to update them progressively.
    current_pred = {k: v.clone() for k, v in init_pred.items()}
    current_pred['joints3d_world'] = joints3d_world.clone()
    optimized_results = {}

    for window in range(window_size, length, window_size):
        if window + window_size >= length:
            window = length
        print(f"\n===== Optimizing frames 1-{window} =====")
        # Slice the data for the current window.
        pred_window = {}
        pred_window['cam'] = current_pred['cam'][:,:window,:]
        pred_window['pose'] = current_pred['pose'][:,:window,:]
        pred_window['betas'] = current_pred['betas'][:,:window,:]
        
        pred_window['trans_world'] = current_pred['trans_world'][:window,:]
        pred_window['poses_root_world'] = current_pred['poses_root_world'][:window,:]

        keypoints_window = keypoints[:window]
        bbox_window = bbox[:,:window,:]
        gt_extrinsics_window = gt_extrinsics[:,:window,:,:]
        joints3d_world_window = current_pred['joints3d_world'][:window]
        
        # Run optimization on the current window.
        optimized_pred_window = custom_smplify.fit(
            pred_window,
            keypoints_window,
            bbox_window,
            gt_extrinsics=gt_extrinsics_window,
            cam_intrinsics=cam_intrinsics,
            joints3d_world=joints3d_world_window
        )

        # Update the current predictions with the optimized values.
        current_pred['cam'][:,:window,:] = optimized_pred_window['cam']
        current_pred['pose'][:,:window,:] = optimized_pred_window['pose']
        current_pred['betas'][:,:window,:] = optimized_pred_window['betas']
        
        current_pred['trans_world'][:window,:] = optimized_pred_window['trans_world']
        current_pred['poses_root_world'][:window,:] = optimized_pred_window['poses_root_world']
        
        current_pred['joints3d_world'][:window] = optimized_pred_window['joints3d_world']

    opt_SMPL_params = custom_smplify.fit_SMPL_params(current_pred, transl_world_aligned, poses_root_world_aligned)

    print('Optimization complete.')

    return opt_SMPL_params

def W_MPJPE_align(cam, bbox, res, cam_intrinsics, smpl, device, pose, betas, transl_wham_world, poses_root_wham_world, gt_extrinsics, window_size):
    n = cam.shape[1]  # number of frames
    # window_size = 100
    transl_world = transl_wham_world.clone()
    poses_root_world = poses_root_wham_world.clone()
    # gt = joblib.load("/mnt/hdd/emdb_dataset/P4/36_outdoor_long_walk/P4_36_outdoor_long_walk_data.pkl")
    joints3d_world = []
    for window in range(0, n, window_size):
        end_window = window + window_size


        if window + window_size >= n:
            end_window = (n - window) + window

        trans_cam = convert_pare_to_full_img_cam(
            cam, 
            bbox[:, :, 2] * 200., 
            bbox[:, :, :2], 
            res[0], 
            res[1], 
            focal_length=cam_intrinsics[:, :, 0, 0])

        # get joints in camera frame [0]
        output = smpl.forward_align(pose[:,window:window+1], betas[:,window:window+1], trans_opt=trans_cam[:,window:window+1].squeeze(0))
        joints3d_cam = output.joints.cpu()

        # get joints in world frame [0]
        output = smpl.forward_align(pose[:,window:window+window_size], betas[:,window:window+window_size], trans_opt=transl_wham_world[window:window+window_size], global_orient_opt=poses_root_wham_world[window:window+window_size])
        joints3d_wham = output.joints.cpu()

        # align joint from wham[0] to cam[0]
        wham_joints_cam, R_wham_cam, t_wham_cam = first_align_joints_return_R_t(joints3d_cam, joints3d_wham)
        R_wham_cam = R_wham_cam.to(device)
        t_wham_cam = t_wham_cam.to(device)
        
        initial_extrinsics = gt_extrinsics.squeeze(0)[window]
        cam_pose = np.linalg.inv(initial_extrinsics.cpu())
        R_cam_pose = torch.tensor(cam_pose[:3, :3]).unsqueeze(0).float().to(device)
        t_cam_pose = torch.tensor(cam_pose[:3, 3]).unsqueeze(0).float().to(device)

        # apply to translation
        sequence_transl_cam = (R_wham_cam @ transl_wham_world[window:end_window].unsqueeze(-1)).squeeze(-1) + t_wham_cam
        sequence_transl_world = (R_cam_pose @ sequence_transl_cam.unsqueeze(-1)).squeeze(-1) + t_cam_pose
        # apply to rotation
        sequence_poses_root_world = R_cam_pose @ R_wham_cam @ poses_root_wham_world[window:end_window]

        transl_world[window:end_window] = sequence_transl_world
        poses_root_world[window:end_window] = sequence_poses_root_world


        wham_joints_world = torch.einsum("tij,tnj->tni", R_cam_pose, wham_joints_cam.to(device)) + t_cam_pose[:, None].to(device)
        joints3d_world.append(wham_joints_world)

    joints3d_world = torch.cat(joints3d_world, dim=0)

    return joints3d_world, transl_world, poses_root_world

