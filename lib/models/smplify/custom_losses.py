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
                #  res,
                 cam_intrinsics,
                 init_pose, 
                 device,
                 gt_extrinsics=None,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.device = device
        # self.res = res
        self.cam_intrinsics = cam_intrinsics
        self.init_pose = init_pose
        self.gt_extrinsics = gt_extrinsics
        
    def forward(self, output, params, input_keypoints, bbox, 
                reprojection_weight=100., regularize_weight=60.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=20.0, sigma=100):
        
        pose, shape, cam, _ , _ = params
        scale = bbox[..., 2:].unsqueeze(-1) * 200.

        # Loss 1. Data term
        pred_keypoints = output[..., :17, :]
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
        cam_diff = compute_jitter(cam).mean() # 0.0
        smooth_error = pose_diff + cam_diff
        
        # Sum up losses
        loss = {
            'reprojection': reprojection_weight * reprojection_error,
            'regularize': regularize_weight * regularize_error,
            'shape': shape_error,
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

            output = smpl.forward_align(params[0], params[1], trans_opt=params[3], global_orient_opt=params[4])

            joints3d = output.joints.unsqueeze(0)

            # get rotation and translation from extrinsics matrix
            rotation = self.gt_extrinsics[:, :, :3, :3]
            translation = self.gt_extrinsics[:, :, :3, 3]
            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )

            len = joints3d.shape[1]
            # make string out of len with leading zeros
            len_str = str(len-1).zfill(5)
            frame = cv2.imread("/mnt/hdd/emdb_dataset/P4/36_outdoor_long_walk/images/" + len_str + ".jpg")
            
            pred_keypoints = full_joints2d[..., :17, :]
             
            # draw keypoints
            # for i in range(pred_keypoints.shape[1]):
            #     x, y = pred_keypoints[0, 0, i].int().tolist()
            #     cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
            loss_dict = self.forward(full_joints2d, params, input_keypoints, bbox)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure