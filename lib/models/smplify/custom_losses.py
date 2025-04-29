import torch

from lib.models.smpl import full_perspective_projection
from lib.eval.eval_utils import first_align_joints_return_R_t
from lib.models.smpl import convert_pare_to_full_img_cam
import numpy as np
import cv2

from lib.utils.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


def gmof(x, sigma=100):
    """
    Geman-McClure error function, cut off at sigma^2
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)

def compute_jitter_custom(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[2:, :] + x[:-2, :] - 2 * x[1:-1, :], dim=-1)

def compute_jitter_velocity(x):
    return torch.norm(x[1:] - x[:-1], dim=-1)

def compute_jitter_combo(x):
    v = torch.norm(x[1:] - x[:-1], dim=-1)
    a = torch.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1)
    return v.mean() + 0.5 * a.mean()

class CustomSMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 res,
                 cam_intrinsics,
                #  init_pose, 
                 device,
                 extrinsics=None,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.device = device
        self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.extrinsics = extrinsics
        
    def forward(self, joints_2d, params, input_keypoints, bbox, init_pred, joint_opt, joints3d_cam=None, joints3d_cam_pred=None,
                reprojection_weight=1., regularize_weight=60.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=200, sigma=100):
        
        scale = bbox[..., 2:].unsqueeze(-1) * 200.

        # Loss 1. Data term
        pred_keypoints = joints_2d[..., :17, :]
        joints_conf = input_keypoints[..., -1:]
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf) / scale).mean()

        # wham_pred_keypoints = joints_2d_wham[..., :17, :]
        # wham_reprojection_error = gmof(wham_pred_keypoints - input_keypoints[..., :-1], sigma)
        # wham_reprojection_error = ((wham_reprojection_error * joints_conf) / scale).mean() 
        
        # Loss 2. Regularization term
        pred_pose_6d = matrix_to_rotation_6d(params[2])
        init_pred_pose_6d = matrix_to_rotation_6d(init_pred['poses_body'])
        # regularize_error = torch.linalg.norm(params[2] - init_pred['poses_body'], dim=-1).mean()
        regularize_error = torch.linalg.norm(pred_pose_6d-init_pred_pose_6d, dim=-1).mean()
        
        # Loss 3. Shape prior and consistency error
        consistency_error = init_pred['betas'].std(dim=1).mean()
        sprior_error = torch.linalg.norm(init_pred['betas'], dim=-1).mean()
        shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        # Loss 4. Smooth loss
        pose_diff = compute_jitter_custom(pred_pose_6d).mean()
        global_orient_diff = compute_jitter_custom(params[1].squeeze(1)).mean()
        trans_diff = compute_jitter_custom(params[0]).mean() # translation in global coords
        # local_trans_diff = compute_jitter(joints3d_cam).mean() #  translation in local coords


        if joint_opt:
            # rotation = rotation_6d_to_matrix(params[3])
            cam_t_diff = compute_jitter_custom(params[4]).mean()

            smooth_error = trans_diff + pose_diff + global_orient_diff + cam_t_diff

            # delta between frames
            vel_pred = params[0][1:,:] - params[0][:-1,:] 
            vel_wham = init_pred['vel_root_world']
            
            # v = torch.tensor([0, 0, 0.1], dtype=torch.float)
            # vel_wham = v.unsqueeze(0).repeat(1226, 1).to(self.device)  # shape [1226, 3]
            
            vel = gmof(vel_pred - vel_wham[:-1], sigma)
            vel_loss = vel.mean()
            #

        else:
            smooth_error = trans_diff + pose_diff + global_orient_diff
            vel_loss = 0


        local_trans_diff = gmof(joints3d_cam - joints3d_cam_pred, sigma).mean() #  translation in local coords


          # Sum up losses
        loss = {
            'reprojection': reprojection_weight * (reprojection_error),# + 10*wham_reprojection_error),
            # 'smooth': smooth_weight * smooth_error,
            # 'velocity': 10000* vel_loss

            # 'regularize': regularize_weight * regularize_error,
            # 'shape': shape_error,
            # 'local': local_trans_diff * 100
            # 'velocity_diff_wham': vel_loss * 1000000,
        }
        
        return loss
        
    def create_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints,
                       init_pred):
        
        def closure():
            optimizer.zero_grad()

            output = smpl.forward_align(params[2], init_pred['betas'], cam_intrinsics=self.cam_intrinsics, 
                                        bbox=bbox, res=self.res, trans_opt=params[0], global_orient_opt=params[1], offset=True)
            joints3d = output.joints.reshape(*init_pred['cam'].shape[:2], -1, 3)

            # get rotation and translation from extrinsics matrix
            rotation = self.extrinsics[:, :, :3, :3]
            translation = self.extrinsics[:, :, :3, 3]
            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )

            # #WHAM keypoints
            # rot_wham = init_pred['wham_cam'][:, :,:3,:3]
            # trans_wham = init_pred['wham_cam'][:, :, :3, 3]
            # wham_joints_2d = full_perspective_projection(
            #     joints3d,
            #     cam_intrinsics=self.cam_intrinsics,
            #     rotation=rot_wham,
            #     translation=trans_wham)




            joints3d_cam = (rotation @ joints3d.transpose(-1, -2)).transpose(-1, -2)
            joints3d_cam = joints3d_cam + translation.unsqueeze(-2)

            trans_cam = convert_pare_to_full_img_cam(
            init_pred['cam'], 
            bbox[:, :, 2] * 200., 
            bbox[:, :, :2], 
            self.res[0], 
            self.res[1], 
            focal_length=self.cam_intrinsics[:, 0, 0])

            # get joints in camera frame [0]
            output = smpl.forward_align(params[2], init_pred['betas'], trans_opt=trans_cam.squeeze(0), 
                                        global_orient_opt=init_pred['poses_root_cam'], offset=False)
            joints3d_cam_pred = output.joints

            loss_dict = self.forward(full_joints2d, params, input_keypoints, bbox, init_pred, joints3d_cam, joints3d_cam_pred) #, wham_joints_2d)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure
    
    def create_cam_smooth_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints,
                       init_pred,
                       joint_opt=False):
        
        def closure():
            optimizer.zero_grad()

            output = smpl.forward_align(params[2], init_pred['betas'], cam_intrinsics=self.cam_intrinsics, 
                                        bbox=bbox, res=self.res, trans_opt=params[0], global_orient_opt=params[1], offset=True)
            joints3d = output.joints.reshape(*init_pred['cam'].shape[:2], -1, 3)

            # get rotation and translation from extrinsics matrix
            rotation = rotation_6d_to_matrix(params[3])
            c = params[4].unsqueeze(-1)
            t = -rotation @ c        
            translation = t.squeeze(-1)  

            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )
            if joint_opt:
                sigma = 100
                pred_keypoints = full_joints2d[..., :17, :]
                joints_conf = input_keypoints[..., -1:]
                reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1])
                reprojection_error = ((reprojection_error * joints_conf) / sigma).mean()

                # smooth_center = compute_jitter_velocity(c).mean()
                # smooth_rot = compute_jitter_velocity(params[3]).mean()
                smooth_center = compute_jitter_custom(c.squeeze(-1)).mean()
                smooth_rot = compute_jitter_custom(params[3]).mean()

                loss = reprojection_error * 5 + (smooth_center + smooth_rot) * 100
                loss.backward()

                return loss
            # preprocessing smoothing
            else:
                sigma = 100
                pred_keypoints = full_joints2d[..., :17, :]
                joints_conf = input_keypoints[..., -1:]
                reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1])
                reprojection_error = ((reprojection_error * joints_conf) / sigma).mean()

                smooth_center = compute_jitter_velocity(c).mean()
                smooth_rot = compute_jitter_velocity(params[3]).mean()
                loss = reprojection_error * 5 + (smooth_center + smooth_rot) * 10
                loss.backward()

                return loss

        return closure
    
    def create_joint_opt_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints,
                       init_pred,
                       joint_opt=False):
        
        def closure():
            optimizer.zero_grad()

            output = smpl.forward_align(params[2], init_pred['betas'], cam_intrinsics=self.cam_intrinsics, 
                                        bbox=bbox, res=self.res, trans_opt=params[0], global_orient_opt=params[1], offset=True)
            joints3d = output.joints.reshape(*init_pred['cam'].shape[:2], -1, 3)

            # get rotation and translation from extrinsics matrix
            # rotation = self.extrinsics[:, :, :3, :3]
            # translation = self.extrinsics[:, :, :3, 3]
            rotation = rotation_6d_to_matrix(params[3])
            c = params[4].unsqueeze(-1)
            t = -rotation @ c        
            translation = t.squeeze(-1)  

            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )
            joints3d_cam = (rotation @ joints3d.transpose(-1, -2)).transpose(-1, -2)
            joints3d_cam = joints3d_cam + translation.unsqueeze(-2)

            trans_cam = convert_pare_to_full_img_cam(
            init_pred['cam'], 
            bbox[:, :, 2] * 200., 
            bbox[:, :, :2], 
            self.res[0], 
            self.res[1], 
            focal_length=self.cam_intrinsics[:, 0, 0])

            # get joints in camera frame [0]
            output = smpl.forward_align(params[2], init_pred['betas'], trans_opt=trans_cam.squeeze(0), 
                                        global_orient_opt=init_pred['poses_root_cam'], offset=False)
            joints3d_cam_pred = output.joints

            loss_dict = self.forward(full_joints2d, params, input_keypoints, bbox, init_pred, joint_opt, joints3d_cam, joints3d_cam_pred) #, wham_joints_2d)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure