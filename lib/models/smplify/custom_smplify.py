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
                 num_steps=30,
                 res=None,
                 device=None,
                 ):
        
        self.smpl = smpl
        self.lr = lr
        self.num_iters = num_iters
        self.num_steps = num_steps
        self.device = device
        self.res = res

    def project_rotation_to_so3(self, R_batch):
        U, _, Vt = torch.linalg.svd(R_batch)
        R_proj = U @ Vt

        # Handle improper rotations (det < 0)
        dets = torch.det(R_proj)
        mask = dets < 0
        U[mask, :, -1] *= -1
        R_proj[mask] = U[mask] @ Vt[mask]

        return R_proj

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
        rot = self.project_rotation_to_so3(rot)

        t = extrinsics.squeeze(0)[ :, :3, 3]
        # c = -rot.transpose(1, 2) @ t.unsqueeze(-1)  # [T, 3, 1]
        c = - torch.linalg.inv(rot) @ t.unsqueeze(-1)  # [T, 3, 1]

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
        
        for j in (j_bar := tqdm(range(20), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            # print(j, msg)
            j_bar.set_postfix_str(msg)

        print(f"Final joint opt loss: {loss.item():.1f}")

        extrinsics = torch.from_numpy(np.eye(4)[None].repeat(pose.shape[0], axis=0)).float().to(self.device)
        Rmat = rotation_6d_to_matrix(params[3].detach())
        Rmat = self.project_rotation_to_so3(Rmat)
        extrinsics[:, :3, :3] = Rmat

        c = params[4].detach().unsqueeze(-1)
        t = (-Rmat @ c).squeeze(-1)
        extrinsics[:, :3, 3] = t
        init_pred['wham_cam'] = extrinsics
        
        return init_pred

    def joint_optimization(self, init_pred, keypoints, bbox, extrinsics, cam_intrinsics, kp_windows, kp_tracks):
        print("Jointly optimizing SMPL and camera extrinsics ...")
        def to_params(param):
            return param.detach().clone().requires_grad_()
        
        pose = matrix_to_rotation_6d(init_pred['poses_body'].clone())
        trans_world = init_pred['trans_world'].clone()
        # poses_root_world = init_pred['poses_root_world'].clone()
        poses_root_world = matrix_to_rotation_6d(init_pred['poses_root_world'].clone())
        
        rot = extrinsics.squeeze(0)[:,:3,:3]
        rot = self.project_rotation_to_so3(rot)

        t = extrinsics.squeeze(0)[ :, :3, 3]
        # c = -rot.transpose(1, 2) @ t.unsqueeze(-1)  # [T, 3, 1]
        c = - torch.linalg.inv(rot) @ t.unsqueeze(-1)  # [T, 3, 1]

        c = c.squeeze(-1).contiguous()
        rot_6d = matrix_to_rotation_6d(rot)

        scale = torch.tensor([1.0]).to(self.device)

        extrinsics_init = extrinsics.squeeze(0).clone()

        extrinsics_rel = compute_frame_relatives(extrinsics.squeeze(0).cpu().numpy())
        extrinsics_rel = torch.from_numpy(extrinsics_rel).float().to(self.device)

        params = [to_params(trans_world), to_params(poses_root_world), to_params(pose), to_params(rot_6d), to_params(c), to_params(scale)]
        # params = [trans_world, poses_root_world, pose, rot_6d, c, scale]
       
        opt_params = [params[0]]
        optimizer = torch.optim.LBFGS(
            opt_params, 
            lr=self.lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        loss_fn = CustomSMPLifyLoss(self.res, cam_intrinsics, device=self.device, extrinsics=extrinsics)
        
        closure = loss_fn.create_joint_opt_closure(optimizer,
                    self.smpl, 
                    params,
                    bbox,
                    keypoints,
                    init_pred,
                    joint_opt=1
                    )
        
        for j in (j_bar := tqdm(range(50), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
        print(f"[1] Loss after opt SMPL translation with WHAM vel: {loss.item():.1f}")

        opt_params = [params[3], params[4], params[5]]

        optimizer = torch.optim.LBFGS(
            opt_params, 
            lr=self.lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        closure = loss_fn.create_joint_opt_closure(optimizer,
                    self.smpl, 
                    params,
                    bbox,
                    keypoints,
                    init_pred,
                    joint_opt=2
                    )
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
        print(f"[1] Loss after opt CAM using reprojection: {loss.item():.1f}")

        params[0] = params[0].detach().clone()
        params[1] = params[1].detach().clone()
        params[2] = params[2].detach().clone()
        params[3] = params[3].detach().clone()
        params[4] = params[4].detach().clone()
        params[5] = params[5].detach().clone()

        print("scale : ", params[5].detach().item())

        #detach params
        trans_world = params[0].clone()
        poses_root_world = params[1].clone()
        pose = params[2].clone()
        rot_6d = params[3].clone()
        c = params[4].clone()
        scale = params[5].clone()

        K = cam_intrinsics.squeeze(0)
        kps0 = kp_tracks[0]
        kps1 = kp_tracks[1]

        extrinsics = torch.from_numpy(np.eye(4)[None].repeat(pose.shape[0], axis=0)).float().to(self.device)
        Rmat = rotation_6d_to_matrix(rot_6d)

        c = c.unsqueeze(-1)
        c_origin = c[0]
        c = scale*(c - c_origin) + c_origin
        t = (-Rmat @ c).squeeze(-1)

        extrinsics[:, :3, 3] = t
        extrinsics[:, :3, :3] = Rmat

        extrinsics_res = extrinsics.clone()

        rel_all = compute_frame_relatives(extrinsics.cpu().numpy())
        rel_all = torch.from_numpy(rel_all).float().to(self.device)

        T_smpl_cam = extrinsics.clone()[:,:3,:3]@trans_world.unsqueeze(-1) + extrinsics.clone()[:,:3,3].unsqueeze(-1)
        T_smpl_cam = T_smpl_cam.squeeze(-1)

        # R_smpl = rotation_6d_to_matrix(poses_root_world).squeeze(1)
        # R_smpl_cam = extrinsics.clone()[:,:3,:3]@R_smpl

        for i, window in enumerate(kp_windows):
            # Get the current window's R and t
            print("Window: ", window)
            start = window[0]
            end = window[1]
            extrinsics_window = extrinsics[start:end+1]
            rel = extrinsics_rel[start:end]

            if end - start > 10:
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

                if error_i < 100:

                    opt_rel = to_params(rel)
                    opt_params = [opt_rel]
                    optimizer = torch.optim.LBFGS(
                        opt_params, 
                        lr=self.lr/2, 
                        max_iter=self.num_iters, 
                        line_search_fn='strong_wolfe')
                    
                    closure_epi = loss_fn.create_epipolar_opt_closure(optimizer,
                        kp0_i,
                        kp1_i,
                        cam_intrinsics,
                        opt_rel,
                        params,
                        self.smpl,
                        init_pred,
                        keypoints,
                        bbox,
                        window,
                        extrinsics[start],
                        extrinsics_init,
                    )
                    
                    for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
                        optimizer.zero_grad()
                        loss = optimizer.step(closure_epi)
                        msg = f'Loss: {loss.item():.1f}'
                        j_bar.set_postfix_str(msg)
                    
                    rel = opt_rel.detach()
                else:
                    print("Error too large, skipping", error_i.item())
            else:
                # If the window is too small, append a large error
                print("Segemtn to small, skipping")

            for k in range(window[0], window[1]):
                extrinsics_res[k+1] = (rel[k - window[0]] @ extrinsics_res[k])

        rot_res = extrinsics_res[:, :3, :3].clone()
        rot_res = self.project_rotation_to_so3(rot_res)
        extrinsics_res[:, :3, :3] = rot_res

        world2_cam = extrinsics_res.clone()
        cam2_world = torch.linalg.inv(world2_cam)
        ones = torch.ones((T_smpl_cam.shape[0], 1), device=T_smpl_cam.device)  # shape [1917, 1]
        T_smpl_cam_hom = torch.cat([T_smpl_cam, ones], dim=1) 
        trans_world = (cam2_world@T_smpl_cam_hom.unsqueeze(-1)).squeeze(-1)

        trans_world = trans_world[:, :3].detach().clone()

        rot = extrinsics_res[:, :3, :3].clone()
        t = extrinsics_res[ :, :3, 3].clone()
        c = - torch.linalg.inv(rot) @ t.unsqueeze(-1)
        c = c.squeeze(-1)
        rot_6d = matrix_to_rotation_6d(rot)

        scale = torch.tensor([1.0]).to(self.device)

        params = [trans_world, poses_root_world, pose, rot_6d, c, scale]
        params[0] = to_params(params[0])

        opt_params = [params[0]]
        optimizer = torch.optim.LBFGS(
            opt_params, 
            lr=self.lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')

        closure = loss_fn.create_joint_opt_closure(optimizer,
                    self.smpl, 
                    params,
                    bbox,
                    keypoints,
                    init_pred,
                    joint_opt=2,
                    )
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
        print(f"[1] Loss after opt translation using reprojection {loss.item():.1f}")
        params[0] = params[0].detach().clone()

        # params[1] = to_params(params[1])
        # opt_params = [params[1]]
        # optimizer = torch.optim.LBFGS(
        #     opt_params, 
        #     lr=self.lr, 
        #     max_iter=self.num_iters, 
        #     line_search_fn='strong_wolfe')
        
        # closure = loss_fn.create_joint_opt_closure(optimizer,
        #             self.smpl, 
        #             params,
        #             bbox,
        #             keypoints,
        #             init_pred,
        #             joint_opt=2
        #             )
        
        # for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
        #     optimizer.zero_grad()
        #     loss = optimizer.step(closure)
        #     msg = f'Loss: {loss.item():.1f}'
        #     j_bar.set_postfix_str(msg)
        # print(f"[1] Loss after opt global orient using reprojection {loss.item():.1f}")
        # params[1] = params[1].detach().clone()

        params[0] = to_params(params[0])
        params[3] = to_params(params[3])
        params[4] = to_params(params[4])
        params[5] = to_params(params[5])

        opt_params = [params[0], params[3], params[4], params[5]]
        optimizer = torch.optim.LBFGS(
            opt_params, 
            lr=self.lr, 
            max_iter=self.num_iters, 
            line_search_fn='strong_wolfe')
        
        closure = loss_fn.create_joint_opt_closure(optimizer,
                    self.smpl, 
                    params,
                    bbox,
                    keypoints,
                    init_pred,
                    joint_opt=3
                    )
        
        for j in (j_bar := tqdm(range(self.num_steps*2), leave=False)):
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
        print(f"Final joint opt loss: {loss.item():.1f}")
        params[0] = params[0].detach().clone()
        params[3] = params[3].detach().clone()
        params[4] = params[4].detach().clone()
        params[5] = params[5].detach().clone()


        # params[1] = to_params(params[1])
        # params[2] = to_params(params[2])
        # opt_params = [params[1], params[2]]
        # optimizer = torch.optim.LBFGS(
        #     opt_params, 
        #     lr=self.lr/2, 
        #     max_iter=self.num_iters, 
        #     line_search_fn='strong_wolfe')
        
        # closure = loss_fn.create_joint_opt_closure(optimizer,
        #             self.smpl, 
        #             params,
        #             bbox,
        #             keypoints,
        #             init_pred,
        #             joint_opt=4
        #             )
        
        # for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
        #     optimizer.zero_grad()
        #     loss = optimizer.step(closure)
        #     msg = f'Loss: {loss.item():.1f}'
        #     j_bar.set_postfix_str(msg)
        # print(f"Final joint opt loss stage 5: {loss.item():.1f}")

        init_pred['trans_world'] = params[0].detach().clone()
        init_pred['poses_root_world'] = rotation_6d_to_matrix(params[1].detach()).clone()
        init_pred['poses_body'] = rotation_6d_to_matrix(params[2].detach()).clone()
        # init_pred['optimized_cam_t'] = extrinsics_res.clone()

        ext = torch.from_numpy(np.eye(4)[None].repeat(pose.shape[0], axis=0)).float().to(self.device)
        Rmat = rotation_6d_to_matrix(params[3].detach())
        Rmat = self.project_rotation_to_so3(Rmat)
        c = params[4].detach().unsqueeze(-1)
        c_origin = c[0]
        scale = params[5].detach()
        print("Scale: ", scale)
        c = scale*(c - c_origin) + c_origin

        t = (-Rmat @ c).squeeze(-1)
        ext[:, :3, 3] = t
        ext[:, :3, :3] = Rmat

        # init_pred['optimized_cam'] = extrinsics_res.clone()
        init_pred['optimized_cam'] = ext.clone()
        
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
    s = 30
    custom_smplify = CustomSMPLify(smpl=smpl, lr=1e-2, num_iters=5, num_steps=s, res=res, device=device)
    
    # get transl and root_pose in gt world frame
    init_pred = W_MPJPE_align_sequentially(init_pred, bbox, res, cam_intrinsics, smpl, device, extrinsics)

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

