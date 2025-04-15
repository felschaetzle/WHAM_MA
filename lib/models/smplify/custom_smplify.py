import os
import torch
from tqdm import tqdm
import numpy as np

from lib.models import build_body_model
from .custom_losses import CustomSMPLifyLoss
from lib.models.smpl import convert_pare_to_full_img_cam
from lib.utils.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
from lib.eval.eval_utils import first_align_joints_return_R_t
from matplotlib import pyplot as plt
import cv2
from lib.models.smpl import full_perspective_projection
from lib.models.smplify.custom_losses import create_SMPL_param_closure

import joblib
class CustomSMPLify():
    
    def __init__(self, 
                 smpl=None,
                 lr=1e-2,
                 num_iters=5,
                 num_steps=10,
                 res=None,
                 device=None,
                 ):
        
        self.smpl = smpl
        self.lr = lr
        self.num_iters = num_iters
        self.num_steps = num_steps
        self.device = device
        self.res = res

    def fit(self, init_pred, keypoints, bbox, extrinsics, cam_intrinsics):
        print("Fitting SMPL parameters...")
        def to_params(param):
            return param.requires_grad_(True)
    

        pose = init_pred['poses_body'].clone()
        transl_world = init_pred['trans_world'].clone()
        poses_root_world = init_pred['poses_root_world'].clone()
        
        params = [to_params(transl_world), to_params(poses_root_world), to_params(pose)]
        # optim_params = [params[0]]
        optimizer = torch.optim.LBFGS(
            params, 
            lr=self.lr, 
            max_iter=self.num_iters)#, 
            # line_search_fn='strong_wolfe')
        
        loss_fn = CustomSMPLifyLoss(self.res, cam_intrinsics, device=self.device, extrinsics=extrinsics)
        
        closure = loss_fn.create_closure(optimizer,
                    self.smpl, 
                    params,
                    bbox,
                    keypoints,
                    init_pred
                    )
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            # print(j, msg)
            j_bar.set_postfix_str(msg)

        print(f"Final joint opt loss: {loss.item():.1f}")

        init_pred['trans_world'] = params[0].detach()
        init_pred['poses_root_world'] = params[1].detach()
        init_pred['poses_body'] = params[2].detach()
        
        return init_pred
    
    def fit_SMPL_params(self, joints3d_world, transl_world, poses_root_world, pose, betas):
        
        def to_params(param):
            return param.requires_grad_(True)
        
        lr = self.lr

        T = torch.eye(4).to(self.device)
        
        params = [to_params(transl_world.clone()), to_params(poses_root_world.clone())]
        # params = [to_params(T)]

        # SMPL param recovery
        optimizer_smpl_params = torch.optim.LBFGS(
            params, 
            lr=lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
                
        closure_smpl_params = create_SMPL_param_closure(optimizer_smpl_params,
                    self.smpl, 
                    params,
                    joints3d_world,
                    pose,
                    betas
                    )
        
        for j in (j_bar := tqdm(range(5), leave=False)):
            optimizer_smpl_params.zero_grad()
            loss = optimizer_smpl_params.step(closure_smpl_params)
            msg = f'Loss: {loss.item():.3f}'
            j_bar.set_postfix_str(msg)
            print(loss.item())

        print(f"Final SMPL param opt loss: {loss.item():.1f}")

        transl_world = params[0].detach()
        poses_root_world = params[1].detach()

        
        return transl_world, poses_root_world
    
# =============================================================================
# Progressive Optimization Function
# =============================================================================
def optimization_upper_bound(init_pred, keypoints, bbox,
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
    s = 50
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
    # window_size = 100
    # window_step = 10
    # get transl and root_pose in gt world frame
    init_pred = W_MPJPE_align_sequentially(init_pred, bbox, res, cam_intrinsics, smpl, device, gt_extrinsics)

    init_pred['trans_world_init'] = init_pred['trans_world'].clone()
    init_pred['poses_root_world_init'] = init_pred['poses_root_world'].clone()
    
    # # Copy the initial predictions to update them progressively.
    # current_pred = {k: v.clone() for k, v in init_pred.items()}
    

    # print("n===== Optimizing frames {window_start}-{window_end} =====")
    # # Slice the data for the current window.
    # pred_window = {}
    # pred_window['cam'] = current_pred['cam'][:,:,:].clone()
    # pred_window['poses_body'] = current_pred['poses_body'][:,:].clone()
    # pred_window['betas'] = current_pred['betas'][:,:,:].clone()
    
    # pred_window['trans_world'] = current_pred['trans_world'][:,:].clone()
    # pred_window['poses_root_world'] = current_pred['poses_root_world'][:,:].clone()
    # pred_window['poses_root_cam'] = current_pred['poses_root_cam'][:,:].clone()

    keypoints_window = keypoints[:]
    bbox_window = bbox[:,:,:]
    gt_extrinsics_window = gt_extrinsics[:,:,:,:]
    
    # Run optimization on the current window.
    optimized_pred_window = custom_smplify.fit(
        init_pred,
        keypoints_window,
        bbox_window,
        extrinsics=gt_extrinsics_window,
        cam_intrinsics=cam_intrinsics
    )

    # Update the current predictions with the optimized values.
    # init_pred['trans_world'] = optimized_pred_window['trans_world']
    # init_pred['poses_root_world'] = optimized_pred_window['poses_root_world']
    # init_pred['poses_body'] = optimized_pred_window['poses_body']

    # current_pred['poses_body'] = rotation_6d_to_matrix(current_pred['poses_body'].reshape(-1, 23, 6))

    print('Optimization complete.')

    return optimized_pred_window




    # gt = joblib.load("/mnt/hdd/emdb_dataset/P9/80_outdoor_walk_big_circle/P9_80_outdoor_walk_big_circle_data.pkl")
    # # gt = joblib.load("/mnt/hdd/emdb_dataset/P4/35_indoor_walk/P4_35_indoor_walk_data.pkl")

    # root_pose = gt['smpl']['poses_root']
    # init_pred['poses_root_world'] = axis_angle_to_matrix(torch.from_numpy(root_pose).float()).unsqueeze(1).to(device)

    # p = axis_angle_to_matrix(torch.from_numpy(gt["smpl"]["poses_body"]).float().reshape(-1, 23, 3))
    # p = matrix_to_rotation_6d(p)
    # init_pred['poses_body'] = p.reshape(-1, 23*6).to(device)


    # init_pred['trans_world'] = torch.from_numpy(gt['smpl']['trans']).float().to(device)

    # noise_std = 0.01  # Standard deviation of the noise
    # noise = torch.randn_like(init_pred['trans_world']) * noise_std
    # init_pred['trans_world'] += noise

    # time_steps = torch.arange(init_pred['trans_world'].shape[0], device=init_pred['trans_world'].device).float()
    # drift_rate = 0.005  # Drift rate per time step
    # drift = drift_rate * time_steps.unsqueeze(-1)  # Shape: [T, 1]
    # init_pred['trans_world'] += drift  # Accumulate drift over time

    # Copy the initial predictions to update them progressively.
    current_pred = {k: v.clone() for k, v in init_pred.items()}
    
    # window_size = length - 1
    b = False
    for window_end in range(window_step, length + window_step, window_step):
        custom_smplify.num_steps = s
        window_start = window_end - window_size
        window_start = max(0, window_start)
        if b:
            break

        if (window_end >= length):# or window_end > 600:
            # b = True
            window_end = length

            custom_smplify.num_steps *= 2
            window_start = 0

        
        print(f"\n===== Optimizing frames {window_start}-{window_end} =====")
        # Slice the data for the current window.
        pred_window = {}
        pred_window['cam'] = current_pred['cam'][:,window_start:window_end,:].clone()
        pred_window['poses_body'] = current_pred['poses_body'][window_start:window_end,:].clone()
        pred_window['betas'] = current_pred['betas'][:,window_start:window_end,:].clone()
        
        pred_window['trans_world'] = current_pred['trans_world'][window_start:window_end,:].clone()
        pred_window['poses_root_world'] = current_pred['poses_root_world'][window_start:window_end,:].clone()
        pred_window['poses_root_cam'] = current_pred['poses_root_cam'][window_start:window_end,:].clone()

        keypoints_window = keypoints[window_start:window_end]
        bbox_window = bbox[:,window_start:window_end,:]
        gt_extrinsics_window = gt_extrinsics[:,window_start:window_end,:,:]
        
        # Run optimization on the current window.
        optimized_pred_window = custom_smplify.fit(
            pred_window,
            keypoints_window,
            bbox_window,
            extrinsics=gt_extrinsics_window,
            cam_intrinsics=cam_intrinsics
        )

        # Update the current predictions with the optimized values.
        current_pred['trans_world'][window_start:window_end] = optimized_pred_window['trans_world']
        current_pred['poses_root_world'][window_start:window_end] = optimized_pred_window['poses_root_world']
        current_pred['poses_body'][window_start:window_end] = optimized_pred_window['poses_body']

    # current_pred['poses_body'] = rotation_6d_to_matrix(current_pred['poses_body'].reshape(-1, 23, 6))

    print('Optimization complete.')

    return current_pred


def test(init_pred, keypoints, bbox, gt_extrinsics, cam_intrinsics,
         smpl, device, length, res):
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
    s = 50
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
    # Copy the initial predictions to update them progressively.
    current_pred = {k: v.clone() for k, v in init_pred.items()}
    
    # current_pred['trans_world'] = gt_extrinsics[]
    rotation = gt_extrinsics[:, :, :3, :3].squeeze(0)
    translation = gt_extrinsics[:, :, :3, 3].squeeze(0)

    c = - rotation.mT @ translation[:,:, None]
    current_pred['trans_world'] = c.squeeze(-1) + rotation[:,:,2] * 1

    window_start = 0
    window_end = 1500

    print(f"\n===== Optimizing frames {window_start}-{window_end} =====")
    # Slice the data for the current window.
    pred_window = {}
    pred_window['cam'] = current_pred['cam'][:,window_start:window_end,:].clone()
    pred_window['poses_body'] = current_pred['poses_body'][window_start:window_end,:].clone()
    pred_window['betas'] = current_pred['betas'][:,window_start:window_end,:].clone()
    
    pred_window['trans_world'] = current_pred['trans_world'][window_start:window_end,:].clone()
    pred_window['poses_root_world'] = current_pred['poses_root_world'][window_start:window_end,:].clone()
    pred_window['poses_root_cam'] = current_pred['poses_root_cam'][window_start:window_end,:].clone()

    keypoints_window = keypoints[window_start:window_end]
    bbox_window = bbox[:,window_start:window_end,:]
    gt_extrinsics_window = gt_extrinsics[:,window_start:window_end,:,:]
    
    # Run optimization on the current window.
    optimized_pred_window = custom_smplify.fit(
        pred_window,
        keypoints_window,
        bbox_window,
        extrinsics=gt_extrinsics_window,
        cam_intrinsics=cam_intrinsics
    )

    # Update the current predictions with the optimized values.
    current_pred['trans_world'][window_start:window_end] = optimized_pred_window['trans_world']
    current_pred['poses_root_world'][window_start:window_end] = optimized_pred_window['poses_root_world']
    current_pred['poses_body'][window_start:window_end] = optimized_pred_window['poses_body']

    print('Optimization complete.')

    return current_pred


def optimization_baseline(init_pred, keypoints, bbox,
                                                  extrinsics, cam_intrinsics,
                                                  smpl, device,
                                                  length, res):
 
    # Create an instance of CustomSMPLify
    s = 50
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
    # window_size = 100
    # window_step = 10

    # get transl and root_pose in gt world frame
    init_pred = W_MPJPE_align_sequentially(init_pred, bbox, res, cam_intrinsics, smpl, device, extrinsics)

    init_pred['trans_world_init'] = init_pred['trans_world'].clone()
    init_pred['poses_root_world_init'] = init_pred['poses_root_world'].clone()

    # Copy the initial predictions to update them progressively.
    current_pred = {k: v.clone() for k, v in init_pred.items()}
    
    optimized_pred_window = custom_smplify.fit(
        current_pred,
        keypoints,
        bbox,
        extrinsics=extrinsics,
        cam_intrinsics=cam_intrinsics
    )
    return optimized_pred_window


    # window_size = length - 1
    b = False
    for window_end in range(window_step, length + window_step, window_step):
        custom_smplify.num_steps = s
        window_start = window_end - window_size
        window_start = max(0, window_start)
        if b:
            break
        # if window_end % 100 == 0:
        #     custom_smplify.num_steps *= 2
        if (window_end >= length): # or window_end > 500:
            # b = True
            window_end = length

            custom_smplify.num_steps *= 2
            window_start = 0

        print(f"\n===== Optimizing frames {window_start}-{window_end} =====")
        # Slice the data for the current window.
        pred_window = {}
        pred_window['cam'] = current_pred['cam'][:,window_start:window_end,:].clone()
        pred_window['poses_body'] = current_pred['poses_body'][window_start:window_end,:].clone()
        pred_window['betas'] = current_pred['betas'][:,window_start:window_end,:].clone()
        
        pred_window['trans_world'] = current_pred['trans_world'][window_start:window_end,:].clone()
        pred_window['poses_root_world'] = current_pred['poses_root_world'][window_start:window_end,:].clone()
        pred_window['poses_root_cam'] = current_pred['poses_root_cam'][window_start:window_end,:].clone()

        keypoints_window = keypoints[window_start:window_end]
        bbox_window = bbox[:,window_start:window_end,:]
        extrinsics_window = extrinsics[:,window_start:window_end,:,:]
        
        # Run optimization on the current window.
        optimized_pred_window = custom_smplify.fit(
            pred_window,
            keypoints_window,
            bbox_window,
            extrinsics=extrinsics_window,
            cam_intrinsics=cam_intrinsics
        )

        # Update the current predictions with the optimized values.
        current_pred['trans_world'][window_start:window_end] = optimized_pred_window['trans_world']
        current_pred['poses_root_world'][window_start:window_end] = optimized_pred_window['poses_root_world']
        current_pred['poses_body'][window_start:window_end] = optimized_pred_window['poses_body']

    print('Optimization complete.')

    return current_pred


def W_MPJPE_align_sequentially(pred, bbox, res, cam_intrinsics, smpl, device, extrinsics, window_size=1):
    # window_size = 20
    print("Aligning joints sequentially, batch size: ", window_size)
    cam = pred['cam']
    pose = pred['poses_body']
    betas = pred['betas']
    transl_wham_world = pred['trans_world']
    poses_root_wham_world = pred['poses_root_world']
    poses_root_cam = pred['poses_root_cam']


    n = cam.shape[1]  # number of frames
    # window_size = n 
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
            focal_length=cam_intrinsics[:, 0, 0])

        # get joints in camera frame [0]
        output = smpl.forward_align(pose[window:window+1], betas[window:window+1], trans_opt=trans_cam[:,window:window+1].squeeze(0),
                                    global_orient_opt=poses_root_cam[window:window+1], offset=False)
        joints3d_cam = output.joints.cpu()

        # get joints in world frame [0]
        output = smpl.forward_align(pose[window:window+window_size], betas[window:window+window_size], 
                                    trans_opt=transl_wham_world[window:window+window_size], 
                                    global_orient_opt=poses_root_wham_world[window:window+window_size], offset=True)
        joints3d_wham = output.joints.cpu()

        # align joint from wham[0] to cam[0]
        wham_joints_cam, R_wham_cam, t_wham_cam = first_align_joints_return_R_t(joints3d_cam, joints3d_wham)
        R_wham_cam = R_wham_cam.to(device)
        t_wham_cam = t_wham_cam.to(device)
        
        initial_extrinsics = extrinsics.squeeze(0)[window]
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

    pred["trans_world"] = transl_world
    pred["poses_root_world"] = poses_root_world

    return pred


def W_MPJPE_align(pred, bbox, res, cam_intrinsics, smpl, device, extrinsics, window_size=None):
    
    # gt_data = joblib.load(gt_data_path)
    pose = pred['poses_body']
    betas = pred['betas']
    cam = pred['cam']
    transl_wham = pred['trans_world']
    poses_root_wham = pred['poses_root_world']
    poses_root_cam = pred['poses_root_cam']

    trans_cam = convert_pare_to_full_img_cam(
        cam, 
        bbox[:, :, 2] * 200., 
        bbox[:, :, :2], 
        res[0], 
        res[1], 
        focal_length=cam_intrinsics[:, 0, 0])

    # get joints in camera frame [0]
    output = smpl.forward_align(pose, betas, trans_opt=trans_cam.squeeze(0), global_orient_opt=poses_root_cam, offset=False)
    joints3d_cam = output.joints.cpu()

    # get joints in world frame [0]
    output = smpl.forward_align(pose, betas, trans_opt=transl_wham, global_orient_opt=poses_root_wham, offset=True)
    joints3d_wham = output.joints.cpu()

    # align joint from wham[0] to cam[0]
    wham_joints_cam, R_wham_cam, t_wham_cam = first_align_joints_return_R_t(joints3d_cam, joints3d_wham)
    R_wham_cam = R_wham_cam.to(device)
    t_wham_cam = t_wham_cam.to(device)
    
    initial_extrinsics = extrinsics[0,0]
    cam_pose = torch.linalg.inv(initial_extrinsics)
    R_cam_pose = cam_pose[:3, :3].unsqueeze(0)
    t_cam_pose = cam_pose[:3, 3].unsqueeze(0)

    # apply to translation
    transl_cam = (R_wham_cam @ transl_wham.unsqueeze(-1)).squeeze(-1) + t_wham_cam
    transl_world = (R_cam_pose @ transl_cam.unsqueeze(-1)).squeeze(-1) + t_cam_pose
    poses_root_world = R_cam_pose @ R_wham_cam @ poses_root_wham

    if "vel_root_world" in pred.keys():
        pred["vel_root_world"] = (R_cam_pose @ R_wham_cam @ pred["vel_root_world"].unsqueeze(-1)).squeeze(-1).squeeze(0)


    wham_joints_world = torch.einsum("tij,tnj->tni", R_cam_pose, wham_joints_cam.to(device)) + t_cam_pose[:, None].to(device)

    # pred['trans_world'] = transl_world
    # pred['poses_root_world'] = poses_root_world
    pred['trans_world'] = transl_world
    pred['poses_root_world'] = poses_root_world
    return pred

