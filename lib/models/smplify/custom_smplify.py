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

    def fit(self, init_pred, keypoints, bbox, gt_extrinsics, cam_intrinsics):
        
        def to_params(param):
            return param.requires_grad_(True)
        
        pose = init_pred['pose']
        betas = init_pred['betas']
        cam = init_pred['cam']

        transl_wham_world = init_pred['trans_world'].squeeze(0)
        poses_root_wham_world = init_pred['poses_root_world'].squeeze(0).unsqueeze(1)
        
        trans_cam = convert_pare_to_full_img_cam(
            cam, 
            bbox[:, :, 2] * 200., 
            bbox[:, :, :2], 
            self.img_w, 
            self.img_h, 
            focal_length=cam_intrinsics[:, :, 0, 0])

        # get joints in camera frame [0]
        output = self.smpl.forward_align(pose, betas, trans_opt=trans_cam.squeeze(0))
        joints3d_cam = output.joints.cpu()

        # get joints in world frame [0]
        output = self.smpl.forward_align(pose, betas, trans_opt=transl_wham_world, global_orient_opt=poses_root_wham_world)
        joints3d_wham = output.joints.cpu()

        # align joint from wham[0] to cam[0]
        wham_joints_cam, R_wham_cam, t_wham_cam = first_align_joints_return_R_t(joints3d_cam, joints3d_wham)
        R_wham_cam = R_wham_cam.to(self.device)
        t_wham_cam = t_wham_cam.to(self.device)
        
        initial_extrinsics = gt_extrinsics.squeeze(0)[0]
        cam_pose = np.linalg.inv(initial_extrinsics.cpu())
        R_cam_pose = torch.tensor(cam_pose[:3, :3]).unsqueeze(0).float().to(self.device)
        t_cam_pose = torch.tensor(cam_pose[:3, 3]).unsqueeze(0).float().to(self.device)

        # apply to translation
        transl_cam = (R_wham_cam @ transl_wham_world.unsqueeze(-1)).squeeze(-1) + t_wham_cam
        transl_world = (R_cam_pose @ transl_cam.unsqueeze(-1)).squeeze(-1) + t_cam_pose
        # apply to rotation
        poses_root_world = R_cam_pose @ R_wham_cam @ poses_root_wham_world

        # keypoints = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)
        transl_world_raw = transl_world.clone()
        BN = pose.shape[1]
        lr = self.lr
        
        # Stage 1. Optimize translation
        params = [to_params(pose), to_params(betas), to_params(cam), to_params(transl_world), to_params(poses_root_world)]
        optim_params = [params[3]]
        
        optimizer = torch.optim.LBFGS(
            optim_params, 
            lr=lr, 
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
            j_bar.set_postfix_str(msg)
            
    
        # # Stage 2. Optimize all params
        # optimizer = torch.optim.LBFGS(
        #     params, 
        #     lr=lr * BN, 
        #     max_iter=self.num_iters, 
        #     line_search_fn='strong_wolfe')
        
        # for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
        #     optimizer.zero_grad()
        #     loss = optimizer.step(closure)
        #     msg = f'Loss: {loss.item():.1f}'
        #     j_bar.set_postfix_str(msg)


        init_pred['pose'] = params[0].detach()
        init_pred['betas'] = params[1].detach()
        init_pred['cam'] = params[2].detach()
        init_pred['trans_world'] = params[3].detach()
        init_pred['poses_root_world'] = params[4].detach()
        init_pred['trans_world_raw'] = transl_world_raw
        
        return init_pred
    

def plot_3d_joints(joints, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
