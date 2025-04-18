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

import joblib

import torch.nn.functional as F
import torch.nn.functional as F

def gaussian_smooth(x, kernel_size=11, sigma=3):
    """
    Apply Gaussian smoothing to a [T, C] tensor over time using reflect padding.
    """
    T, C = x.shape

    # Make Gaussian kernel
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, device=x.device).float()
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).repeat(C, 1, 1)  # [C, 1, K]

    # Reshape x to [1, C, T]
    x = x.T.unsqueeze(0)  # [1, C, T]

    # Pad with reflect mode
    x_padded = F.pad(x, (half, half), mode='reflect')  # [1, C, T + 2*half]

    # Apply per-channel 1D convolution (depthwise)
    x_smooth = F.conv1d(x_padded, kernel, groups=C)  # [1, C, T]

    return x_smooth.squeeze(0).T  # [T, C]

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
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
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
    
    def smooth_extrinsics(self, init_pred, keypoints, bbox, extrinsics, cam_intrinsics):
        print("Smooth extrinsics")
        def to_params(param):
            return param.requires_grad_(True)
    
        pose = init_pred['poses_body'].clone()
        transl_world = init_pred['trans_world'].clone()
        poses_root_world = init_pred['poses_root_world'].clone()

        rot = extrinsics.squeeze(0)[:,:3,:3]
        t = extrinsics.squeeze(0)[ :, :3, 3]
        c = -rot.transpose(1, 2) @ t.unsqueeze(-1)  # [T, 3, 1]
        c = c.squeeze(-1)  

        c = gaussian_smooth(c, kernel_size=21, sigma=5)

        rot_6d = matrix_to_rotation_6d(rot)

        params = [transl_world, poses_root_world, pose, to_params(rot_6d), to_params(c.contiguous())]
        optim_params = [params[3], params[4]]
        optimizer = torch.optim.LBFGS(
            optim_params, 
            lr=self.lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        loss_fn = CustomSMPLifyLoss(self.res, cam_intrinsics, device=self.device)#, extrinsics=extrinsics)
        
        closure = loss_fn.create_cam_smooth_closure(optimizer,
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

        extrinsics = torch.from_numpy(np.eye(4)[None].repeat(pose.shape[0], axis=0)).float().to(self.device)
        Rmat = rotation_6d_to_matrix(params[3].detach())
        extrinsics[:, :3, :3] = Rmat

        c = params[4].detach().unsqueeze(-1)
        t = (-Rmat @ c).squeeze(-1)
        extrinsics[:, :3, 3] = t
        init_pred['wham_cam'] = extrinsics
        
        return init_pred
    
def optimization_upper_bound(init_pred, keypoints, bbox,
                                                  gt_extrinsics, cam_intrinsics,
                                                  smpl, device,
                                                  length, res):
  
    # Create an instance of CustomSMPLify
    s = 50
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
    # get transl and root_pose in gt world frame
    init_pred = W_MPJPE_align_sequentially(init_pred, bbox, res, cam_intrinsics, smpl, device, gt_extrinsics)

    init_pred['trans_world_init'] = init_pred['trans_world'].clone()
    init_pred['poses_root_world_init'] = init_pred['poses_root_world'].clone()
    
    # Run optimization on the current window.
    optimized_pred_window = custom_smplify.fit(
        init_pred,
        keypoints,
        bbox,
        extrinsics=gt_extrinsics,
        cam_intrinsics=cam_intrinsics
    )

    print('Optimization complete.')

    return optimized_pred_window

def optimization_baseline(init_pred, keypoints, bbox,
                                                  extrinsics, cam_intrinsics,
                                                  smpl, device,
                                                  length, res):
 
    # Create an instance of CustomSMPLify
    s = 50
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
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

"""    b = False
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
"""

def W_MPJPE_align_sequentially(pred, bbox, res, cam_intrinsics, smpl, device, extrinsics, window_size=1):
    
    # window_size = 20
    print("Aligning joints sequentially, batch size: ", window_size)
    cam = pred['cam']
    pose = pred['poses_body']
    betas = pred['betas']
    transl_wham_world = pred['trans_world']
    poses_root_wham_world = pred['poses_root_world']
    poses_root_cam = pred['poses_root_cam']


    n = cam.shape[1]
    transl_world = transl_wham_world.clone()
    poses_root_world = poses_root_wham_world.clone()
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

    pred['trans_world'] = transl_world
    pred['poses_root_world'] = poses_root_world
    return pred

