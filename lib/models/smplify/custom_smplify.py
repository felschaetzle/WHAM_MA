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

from scripts.superglue_tracker import get_fundamental_matrix_torch, compute_epipolar_lines_batch_torch, epipolar_distances_batch_torch
from scripts.extrinsics_classifier import compute_frame_relatives


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

        c = gaussian_smooth(c, kernel_size=51, sigma=10)

        scale = torch.tensor([1.0]).to(self.device)

        rot_6d = matrix_to_rotation_6d(rot)

        params = [transl_world, poses_root_world, pose, to_params(rot_6d), to_params(c.contiguous()), scale]
        optim_params = [params[3]]
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
                    init_pred,
                    opt=1
                    )
        
        for j in (j_bar := tqdm(range(2), leave=False)):
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

    def joint_optimization(self, init_pred, keypoints, bbox, extrinsics, cam_intrinsics, kp_windows, kp_tracks):
        print("Jointly optimizing SMPL and camera extrinsics ...")
        def to_params(param):
            return param.requires_grad_(True)
    
        pose = init_pred['poses_body'].clone()
        transl_world = init_pred['trans_world'].clone()
        # poses_root_world = init_pred['poses_root_world'].clone()
        poses_root_world = matrix_to_rotation_6d(init_pred['poses_root_world'].clone())
        
        # rot = extrinsics.squeeze(0)[:,:3,:3]
        # t = extrinsics.squeeze(0)[ :, :3, 3]
        # c = -rot.transpose(1, 2) @ t.unsqueeze(-1)  # [T, 3, 1]
        # c = c.squeeze(-1).contiguous()
        # rot_6d = matrix_to_rotation_6d(rot)

        scale = torch.tensor([1.0]).to(self.device)

        extrinsics_rel = compute_frame_relatives(extrinsics.squeeze(0).cpu().numpy())
        extrinsics_rel = torch.from_numpy(extrinsics_rel).float().to(self.device)

        params = [to_params(transl_world), to_params(poses_root_world), to_params(pose), to_params(extrinsics_rel), extrinsics[0,0], to_params(scale)]

        extrinsics_res = extrinsics.squeeze(0).clone()
        
        # opt_params = [params[0]]
        # optimizer = torch.optim.LBFGS(
        #     opt_params, 
        #     lr=self.lr, 
        #     max_iter=self.num_iters, 
        #     line_search_fn='strong_wolfe')
        
        loss_fn = CustomSMPLifyLoss(self.res, cam_intrinsics, device=self.device, extrinsics=extrinsics)
        
        # closure = loss_fn.create_joint_opt_closure(optimizer,
        #             self.smpl, 
        #             params,
        #             bbox,
        #             keypoints,
        #             init_pred,
        #             kp_windows,
        #             kp_tracks,
        #             joint_opt=1
        #             )
        
        # for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
        #     optimizer.zero_grad()
        #     loss = optimizer.step(closure)
        #     msg = f'Loss: {loss.item():.1f}'
        #     j_bar.set_postfix_str(msg)

        # print(f"Final joint opt loss stage 1: {loss.item():.1f}")

        # opt_params = [params[3], params[4]]

        # optimizer = torch.optim.LBFGS(
        #     opt_params, 
        #     lr=self.lr, 
        #     max_iter=self.num_iters, 
        #     line_search_fn='strong_wolfe')
        
        # closure_2 = loss_fn.create_joint_opt_closure(optimizer,
        #             self.smpl, 
        #             params,
        #             bbox,
        #             keypoints,
        #             init_pred,
        #             kp_windows,
        #             kp_tracks,
        #             joint_opt=2
        #             )
        
        # for j in (j_bar := tqdm(range(10), leave=False)):
        #     optimizer.zero_grad()
        #     loss = optimizer.step(closure_2)
        #     msg = f'Loss: {loss.item():.1f}'
        #     j_bar.set_postfix_str(msg)
        
        # print(f"Final joint opt loss stage 2: {loss.item():.1f}")

        # opt_params = [params[0], params[3], params[4], params[5]]

        # optimizer = torch.optim.LBFGS(
        #     opt_params, 
        #     lr=self.lr, 
        #     max_iter=self.num_iters, 
        #     line_search_fn='strong_wolfe')
        
        # closure_3 = loss_fn.create_joint_opt_closure(optimizer,
                    # self.smpl, 
                    # params,
                    # bbox,
                    # keypoints,
                    # init_pred,
        #             kp_windows,
        #             kp_tracks,
        #             joint_opt=3
        #             )
        
        # for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
        #     optimizer.zero_grad()
        #     loss = optimizer.step(closure_3)
        #     msg = f'Loss: {loss.item():.1f}'
        #     j_bar.set_postfix_str(msg)
        
        # print(f"Final joint opt loss stage 3: {loss.item():.1f}")

        K = cam_intrinsics.squeeze(0)
        kps0 = kp_tracks[0]
        kps1 = kp_tracks[1]

        # rel_all = compute_frame_relatives(extrinsics.squeeze(0).cpu().numpy())
        # rel_all = torch.from_numpy(rel_all).float().to(self.device)

        for i, window in enumerate(kp_windows):
            # Get the current window's R and t
            print("Window: ", window)
            start = window[0]
            end = window[1]
            extrinsics_window = extrinsics.squeeze(0)[start:end+1]

            if end - start > 10:
                # rel = extrinsics_rel[start:end]
                T_start = extrinsics_window[0]
                T_end = extrinsics_window[-1]

                # Get the current window's keypoints
                kp0_i = kps0[i]
                kp1_i = kps1[i]

                kp0_i = torch.from_numpy(kp0_i).float().to(self.device)
                kp1_i = torch.from_numpy(kp1_i).float().to(self.device)

                F = get_fundamental_matrix_torch(K, T_end, T_start)
                lines = compute_epipolar_lines_batch_torch(F, kp0_i)
                epi_error = epipolar_distances_batch_torch(lines, kp1_i)

                quantile = 50/100

                thresh   = torch.quantile(epi_error, quantile)                            # median
                epi_keep  = epi_error[epi_error <= thresh]

                error_i = epi_keep.mean()

                if error_i < 50:

                    # params = [to_params(transl_world), to_params(poses_root_world), to_params(pose), to_params(rel), extrinsics[0,0], to_params(scale)]
                    opt_params = [params[0], params[3]]
                    optimizer = torch.optim.LBFGS(
                        opt_params, 
                        lr=self.lr, 
                        max_iter=self.num_iters, 
                        line_search_fn='strong_wolfe')
                    
                    closure_epi = loss_fn.create_epipolar_opt_closure(optimizer,
                        self.smpl, 
                        params,
                        bbox,
                        keypoints,
                        init_pred,
                        kp0_i,
                        kp1_i,
                        cam_intrinsics,
                        window
                    )
                    
                    for j in (j_bar := tqdm(range(5), leave=False)):
                        optimizer.zero_grad()
                        loss = optimizer.step(closure_epi)
                        msg = f'Loss: {loss.item():.1f}'
                        j_bar.set_postfix_str(msg)
                    
                    # rel = opt_params[0].detach()
                else:
                    print("Error too large, skipping", error_i.item())
            else:
                # If the window is too small, append a large error
                print("Segemtn to small, skipping")

            for k in range(window[0], window[1]):
                    # print(k, extrinsics[0,k+1].shape, rel[k - window[0]].shape, extrinsics[0,k].shape)
                extrinsics_res[k+1] = extrinsics_rel[k].detach() @ extrinsics_res[k]

        init_pred['optimized_cam'] = extrinsics_res .detach().squeeze(0)

        init_pred['trans_world'] = params[0].detach()
        init_pred['poses_root_world'] = rotation_6d_to_matrix(params[1].detach())
        init_pred['poses_body'] = params[2].detach()

        # ext = torch.from_numpy(np.eye(4)[None].repeat(pose.shape[0], axis=0)).float().to(self.device)
        # Rmat = rotation_6d_to_matrix(params[3].detach())
        # ext[:, :3, :3] = Rmat

        # c = params[4].detach().unsqueeze(-1)
        # c_origin = c[0]
        # scale = params[5].detach()
        # print("Scale: ", scale)
        # c = scale*(c - c_origin) + c_origin

        # t = (-Rmat @ c).squeeze(-1)
        # ext[:, :3, 3] = t

        # scale = params[5].detach()
        # T_rel = params[3].detach()
    
        # T0 = params[4].detach()

        # Ts = [T0]

        # for i in range(T_rel.shape[0]):
        #     # note: this is out-of-place
        #     Ts.append(T_rel[i] @ Ts[-1])

        # # now Ts is a list of length N+1, each [4,4]
        # T = torch.stack(Ts, dim=0)        # [N+1,4,4]

        # rotation = T[:, :3, :3]
        # translation = T[:, :3, 3]
        # c = (-rotation @ translation.unsqueeze(-1)).squeeze(-1)
        # c_origin = c[0]
        # c = scale*(c - c_origin) + c_origin

        # t = -rotation @ c.unsqueeze(-1) 
        # T[:, :3, 3] = t.squeeze(-1)

        # init_pred['optimized_cam'] = T
        
        return init_pred
        
def optimization_upper_bound(init_pred, keypoints, bbox,
                                                  gt_extrinsics, cam_intrinsics,
                                                  smpl, device,
                                                  length, res):
  
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
                                                  length, res, kp_windows, kp_tracks):
 
    # Create an instance of CustomSMPLify
    s = 10
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
    # get transl and root_pose in gt world frame
    # init_pred = W_MPJPE_align_sequentially(init_pred, bbox, res, cam_intrinsics, smpl, device, extrinsics)

    init_pred['trans_world_init'] = init_pred['trans_world'].clone()
    init_pred['poses_root_world_init'] = init_pred['poses_root_world'].clone()
    
    # Copy the initial predictions to update them progressively.
    current_pred = {k: v.clone() for k, v in init_pred.items()}
    
    optimized_pred_window = custom_smplify.joint_optimization(
        current_pred,
        keypoints,
        bbox,
        extrinsics,
        cam_intrinsics,
        kp_windows,
        kp_tracks
    )

    return optimized_pred_window

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

