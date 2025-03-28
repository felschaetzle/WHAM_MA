import torch

from lib.models.smpl import full_perspective_projection
from lib.eval.eval_utils import first_align_joints_return_R_t
from lib.models.smpl import convert_pare_to_full_img_cam
import numpy as np
import cv2


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """
    Compute jitter for the input tensor
    """
    return torch.linalg.norm(x[:, 2:] + x[:, :-2] - 2 * x[:, 1:-1], dim=-1)


class CustomSMPLifyLoss(torch.nn.Module):
    def __init__(self, 
                 res,
                 cam_intrinsics,
                 init_pose, 
                 device,
                 gt_extrinsics=None,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.device = device
        self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.init_pose = init_pose
        self.gt_extrinsics = gt_extrinsics
        
    def forward(self, joints_2d, params, input_keypoints, bbox, joints_3d,
                reprojection_weight=100., regularize_weight=60.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=20.0, sigma=100):
        
        pose, shape, cam, _ , _ = params
        scale = bbox[..., 2:].unsqueeze(-1) * 200.

        # Loss 1. Data term
        pred_keypoints = joints_2d[..., :17, :]
        joints_conf = input_keypoints[..., -1:]
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf) / scale).mean()
        
        # Loss 2. Regularization term
        regularize_error = torch.linalg.norm(pose - self.init_pose, dim=-1).mean()
        
        # Loss 3. Shape prior and consistency error
        consistency_error = shape.std(dim=1).mean()
        sprior_error = torch.linalg.norm(shape, dim=-1).mean()
        shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        # Loss 4. Smooth loss
        pose_diff = compute_jitter(pose).mean()
        # cam_diff = compute_jitter(cam).mean() # 0.0
        trans_diff = compute_jitter(params[3]).mean() # translation in global coords
        smooth_error = pose_diff + trans_diff/10
        
        # Sum up losses
        loss = {
            'reprojection': reprojection_weight * reprojection_error,
            'regularize': regularize_weight * regularize_error,
            # 'shape': shape_error,
            'smooth': smooth_weight * smooth_error
        }
        
        return loss
        
    def create_closure(self,
                       optimizer,
                       smpl, 
                       params,
                       bbox,
                       input_keypoints):
        
        def closure():
            optimizer.zero_grad()

            output = smpl.forward_align(params[0], params[1], cam_intrinsics=self.cam_intrinsics, bbox=bbox, res=self.res, trans_opt=params[3], global_orient_opt=params[4])
            joints3d = output.joints.reshape(*params[2].shape[:2], -1, 3)

            # get rotation and translation from extrinsics matrix
            rotation = self.gt_extrinsics[:, :, :3, :3]
            translation = self.gt_extrinsics[:, :, :3, 3]
            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )

            loss_dict = self.forward(full_joints2d, params, input_keypoints, bbox, joints3d)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure
    
def create_SMPL_param_closure(optimizer, smpl, params, joints3d, pose, betas):
    
    def closure():
        optimizer.zero_grad()

        # T = params[0]
        # T = T.unsqueeze(0).expand(transl.shape[0], -1, -1)
        # # transform transl and global_orient from wham to world using T
        # transl_world = torch.matmul(T[:, :3, :3], transl.unsqueeze(-1)).squeeze(-1) + T[:, :3, 3]
        # global_orient_world = torch.matmul(T[:, :3, :3].unsqueeze(1), global_orient)



        output = smpl.forward_align(pose, betas, trans_opt=params[0], global_orient_opt=params[1], offset=True)
        pred_joints3d = output.joints[:, :17, :]

        # Calculate 3D distance between predicted and GT joints
        loss = torch.linalg.norm(pred_joints3d[0, 0, :] - joints3d[0, 0, :]).mean()  # Ensure loss is a scalar
        # print("loss: ", loss)
        loss.backward()
        return loss
    
    return closure
