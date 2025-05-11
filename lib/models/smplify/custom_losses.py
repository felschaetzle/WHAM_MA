import torch

from lib.models.smpl import full_perspective_projection
from lib.eval.eval_utils import first_align_joints_return_R_t
from lib.models.smpl import convert_pare_to_full_img_cam
import numpy as np
import cv2

from lib.utils.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
# from scripts.extrinsics_classifier import compute_epipolar_lines_batch, epipolar_distances_batch
from scripts.superglue_tracker import get_fundamental_matrix_torch, compute_epipolar_lines_batch_torch, epipolar_distances_batch_torch, get_fundamental_matrix_torch_relative

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

# def compute_jitter_combo(x):
#     v = torch.norm(x[1:] - x[:-1], dim=-1)
#     a = torch.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1)
#     return v.mean() + 0.5 * a.mean()

# copmpute epipolar error given a set of 2d tracks for a set of windows in one sequence. Each window has one R,t and 2kps
def compute_epipolar_error(rot, trans, K, kp_tracks, kp_windows):
    """
    Compute the epipolar error for a set of 2D keypoints in a sequence.
    """
    # Get the number of windows
    K = K.squeeze(0)
    kps0 = kp_tracks[0]
    kps1 = kp_tracks[1]
    # Initialize the error list
    errors = []

    for i, window in enumerate(kp_windows):
        # Get the current window's R and t

        start = window[0]
        end = window[1]

        if end - start > 10:
            r_start = rot[start]
            trans_start = trans[start]

            r_end = rot[end]
            trans_end = trans[end]

            T_start = torch.eye(4).to(rot.device).float()
            T_start[:3, :3] = r_start
            T_start[:3, 3] = trans_start

            T_end = torch.eye(4).to(rot.device).float()
            T_end[:3, :3] = r_end
            T_end[:3, 3] = trans_end

            # Get the current window's keypoints
            kp0_i = kps0[i]
            kp1_i = kps1[i]

            kp0_i = torch.from_numpy(kp0_i).float().to(rot.device)
            kp1_i = torch.from_numpy(kp1_i).float().to(rot.device)

            F = get_fundamental_matrix_torch(K, T_end, T_start)
            lines = compute_epipolar_lines_batch_torch(F, kp0_i)
            epi_error = epipolar_distances_batch_torch(lines, kp1_i)

            quantile = 30/100

            thresh   = torch.quantile(epi_error, quantile)                            # median
            epi_keep  = epi_error[epi_error <= thresh]

            error_i = epi_keep.mean()

            if error_i < 100:
                errors.append(error_i)
            else:
                errors.append(torch.tensor(0.0, device=rot.device, dtype=float))
        else:
            # If the window is too small, append a large error
            errors.append(torch.tensor(0.0, device=rot.device, dtype=float))
    errors = torch.stack(errors)  # shape [N]
    return errors.sum()

def compute_epipolar_error_relative(T_rel, K, kp0_i, kp1_i):

    T = torch.eye(4).float().to(T_rel.device)
    for i in range(T_rel.shape[0]):
        T = T_rel[i] @ T

    F = get_fundamental_matrix_torch_relative(K, T)
    lines = compute_epipolar_lines_batch_torch(F, kp0_i)
    epi_error = epipolar_distances_batch_torch(lines, kp1_i)

    quantile = 30/100

    thresh   = torch.quantile(epi_error, quantile)                            # median
    epi_keep  = epi_error[epi_error <= thresh]

    error_i = epi_keep.mean()
    return error_i

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
        
    def forward(self, joints_2d, params, input_keypoints, bbox, init_pred, joint_opt, c=None, joints3d_cam=None, joints3d_cam_pred=None,
                reprojection_weight=1., regularize_weight=60.0, 
                consistency_weight=10.0, sprior_weight=0.04, 
                smooth_weight=100, sigma=100):
        
        scale = bbox[..., 2:].unsqueeze(-1) * 200.

        # Loss 1. Data term
        pred_keypoints = joints_2d[..., :17, :]
        joints_conf = input_keypoints[..., -1:]
        reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1], sigma)
        reprojection_error = ((reprojection_error * joints_conf) / scale).mean()

        # Loss 2. Regularization term
        # pred_pose_6d = matrix_to_rotation_6d()
        # init_pred_pose_6d = matrix_to_rotation_6d(init_pred['poses_body'])
        # regularize_error = torch.linalg.norm(params[2] - init_pred['poses_body'], dim=-1).mean()
        # regularize_error = torch.linalg.norm(pred_pose_6d-init_pred_pose_6d, dim=-1).mean()
        
        # Loss 3. Shape prior and consistency error
        # consistency_error = init_pred['betas'].std(dim=1).mean()
        # sprior_error = torch.linalg.norm(init_pred['betas'], dim=-1).mean()
        # shape_error = sprior_weight * sprior_error + consistency_weight * consistency_error
        
        # Loss 4. Smooth loss
        pose_diff = compute_jitter_custom(params[2]).mean()
        global_orient_diff = compute_jitter_custom(params[1].squeeze(1)).mean()

        if joint_opt is not None:

            if joint_opt == 1:

                """             
                v = torch.tensor([0, 0, 0.1], dtype=torch.float)
                vel_wham = v.unsqueeze(0).repeat(1226, 1).to(self.device)  # shape [1226, 3]

                delta between frames [THIS IS A WRONG WAY TO DO IT]
                vel_pred = params[0][1:,:] - params[0][:-1,:] 
                vel_wham = init_pred['vel_root_world']
                vel = gmof(vel_pred - vel_wham[:-1], sigma)
                vel_loss = vel.mean()

                norm_vel_wham = vel_wham.norm(dim=1)
                norm_vel_pred = vel_pred.norm(dim=1)
                norm_vel = gmof(norm_vel_pred - norm_vel_wham[:-1], sigma)
                vel_loss = norm_vel.mean()
                """
                
                vel_body_wham = init_pred['vel_root']
                vel_pred_world = params[0][1:,:] - params[0][:-1,:]
                global_orient = params[1].squeeze(1)
                body_to_world = rotation_6d_to_matrix(global_orient)
                world_to_body = body_to_world.inverse()
                vel_pred_body = torch.einsum('ijk,ik->ij', world_to_body[:-1], vel_pred_world)
                vel = gmof(vel_pred_body - vel_body_wham[:-1])
                vel_loss = vel.mean()
                
                # gt = joblib.load("/mnt/hdd/emdb_dataset/P4/35_indoor_walk/P4_35_indoor_walk_data.pkl")

                loss = {
                    'velocity': 100000* vel_loss,
                }

            elif joint_opt == 2:
                loss = {
                    'reprojection': reprojection_weight * (reprojection_error)
                }


            elif joint_opt == 3:
                # rotation = rotation_6d_to_matrix(params[3])
                trans_diff = compute_jitter_custom(params[0]).mean() # translation in global coords

                cam_c_diff = compute_jitter_custom(c).mean()
                cam_R_diff = compute_jitter_custom(params[3]).mean()

                smooth_error = trans_diff + cam_c_diff + cam_R_diff 
                
                vel_body_wham = init_pred['vel_root']
                vel_pred_world = params[0][1:,:] - params[0][:-1,:]
                global_orient = params[1].squeeze(1)
                body_to_world = rotation_6d_to_matrix(global_orient)
                world_to_body = body_to_world.inverse()
                vel_pred_body = torch.einsum('ijk,ik->ij', world_to_body[:-1], vel_pred_world)
                vel = gmof(vel_pred_body - vel_body_wham[:-1])
                vel_loss = vel.mean()
                
                loss = {
                    'reprojection': reprojection_weight * (reprojection_error),
                    'smooth': smooth_weight * smooth_error,
                    'velocity': 10000* vel_loss,
                }

            elif joint_opt == 4:

                smooth_error = global_orient_diff + pose_diff
                
                loss = {
                    'reprojection': reprojection_weight * (reprojection_error),
                    'smooth': smooth_weight * smooth_error,
                    # 'velocity': 10000* vel_loss,
                }
        else:
            trans_diff = compute_jitter_custom(params[0]).mean() # translation in global coords

            smooth_error = trans_diff + pose_diff + global_orient_diff
            # Sum up losses
            loss = {
                'reprojection': reprojection_weight * (reprojection_error),
                'smooth': smooth_weight * smooth_error
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
                       opt=False):
        
        def closure():
            optimizer.zero_grad()

            output = smpl.forward_align(params[2], init_pred['betas'], cam_intrinsics=self.cam_intrinsics, 
                                        bbox=bbox, res=self.res, trans_opt=params[0], global_orient_opt=params[1], offset=True)
            joints3d = output.joints.reshape(*init_pred['cam'].shape[:2], -1, 3)

            # get rotation and translation from extrinsics matrix
            rotation = rotation_6d_to_matrix(params[3])
            c = params[4].unsqueeze(-1)

            c_origin = c[0]
            scale = params[5]
            c = scale*(c - c_origin) + c_origin

            t = -rotation @ c        
            translation = t.squeeze(-1)  

            # translation = ext[:, :3, 3]
            # rotation = ext[:, :3, :3]


            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )
            
            sigma = 100
            pred_keypoints = full_joints2d[..., :17, :]
            joints_conf = input_keypoints[..., -1:]
            reprojection_error = gmof(pred_keypoints - input_keypoints[..., :-1])
            reprojection_error = ((reprojection_error * joints_conf) / sigma).mean()

            # smooth_center = compute_jitter_velocity(c).mean()
            smooth_rot = compute_jitter_velocity(params[3]).mean()
            loss = reprojection_error  + (smooth_rot)*100
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
                       joint_opt=None
                       ):
        
        def closure():
            optimizer.zero_grad()



            output = smpl.forward_align(params[2], init_pred['betas'], cam_intrinsics=self.cam_intrinsics, 
                                        bbox=bbox, res=self.res, trans_opt=params[0], global_orient_opt=params[1], offset=True)
            joints3d = output.joints.reshape(*init_pred['cam'].shape[:2], -1, 3)

            scale = params[5]

            rotation = rotation_6d_to_matrix(params[3])
            c = params[4].unsqueeze(-1)
            c_origin = c[0]
            scale = params[5]
            c = scale*(c - c_origin) + c_origin

            t = -rotation @ c        
            translation = t.squeeze(-1)  

            full_joints2d = full_perspective_projection(
                joints3d,
                cam_intrinsics=self.cam_intrinsics,
                rotation=rotation,
                translation=translation,
            )

            loss_dict = self.forward(full_joints2d, params, input_keypoints, bbox, init_pred, joint_opt, c=c)
            loss = sum(loss_dict.values())
            loss.backward()
            return loss
        
        return closure
    

    def create_epipolar_opt_closure(self,
                        optimizer,
                        kp0_i,
                        kp1_i,
                        cam_intrinsics,
                        opt_rel,
                        params,
                        smpl,
                        init_pred,
                        input_keypoints,
                        bbox,
                        window,
                        extrinsics_start,
                        extrinsics_initial,
                        # trans_window
                        ):
        
        def closure():
            optimizer.zero_grad()

            # trans_window = params[0][window[0]:window[1]+1]
            # global_orient_window = params[1][window[0]:window[1]+1]
            # pose_window = params[2][window[0]:window[1]+1]
            # betas_window = init_pred["betas"][window[0]:window[1]+1]
            # bbox_window = bbox[:,window[0]:window[1]+1]
            # input_keypoints_window = input_keypoints[window[0]:window[1]+1]

            # output = smpl.forward_align(pose_window, betas_window, cam_intrinsics=self.cam_intrinsics, 
            #                             bbox=bbox_window, res=self.res, trans_opt=trans_window, global_orient_opt=global_orient_window, offset=True)
            # joints3d = output.joints.reshape(1, trans_window.shape[0], -1, 3)

            # output = smpl.forward_align(pose_window, betas_window, cam_intrinsics=self.cam_intrinsics, 
            #                             bbox=bbox_window, res=self.res, trans_opt=init_pred['trans_world'][window[0]:window[1]+1], global_orient_opt=global_orient_window, offset=True)
            # joints3d_init = output.joints.reshape(1, trans_window.shape[0], -1, 3)

            # scale = params[5]

            # Ts = [extrinsics_start]

            # for i in range(opt_rel.shape[0]):
            #     Ts.append(opt_rel[i] @ Ts[-1])
            # Ts = torch.stack(Ts)

            # # local joints
            # J = joints3d.squeeze(0)

            # # make homogeneous coords: [89,31,4]
            # ones = torch.ones(J.shape[0], J.shape[1], 1, device=J.device, dtype=J.dtype)
            # J_hom = torch.cat([J, ones], dim=-1)   # [89,31,4]

            # # for batch‐matmul we need shape [89,4,31]
            # J_hom_t = J_hom.transpose(1,2)         # [89,4,31]

            # # Ts: [89,4,4]; batch matmul → [89,4,31]
            # J_tr_hom_t = Ts @ J_hom_t              # [89,4,31]

            # # back to [89,31,4]
            # J_tr_hom = J_tr_hom_t.transpose(1,2)   # [89,31,4]

            # # drop the homogeneous 1s → [89,31,3]
            # J_local = J_tr_hom[..., :3]

            # J_init = joints3d_init.squeeze(0)

            # # make homogeneous coords: [89,31,4]
            # ones = torch.ones(J_init.shape[0], J_init.shape[1], 1, device=J_init.device, dtype=J_init.dtype)
            # J_hom_init = torch.cat([J_init, ones], dim=-1)   # [89,31,4]

            # # for batch‐matmul we need shape [89,4,31]
            # J_hom_init_t = J_hom_init.transpose(1,2)         # [89,4,31]

            # # Ts: [89,4,4]; batch matmul → [89,4,31]
            # J_tr_hom_init_t = extrinsics_initial[window[0]:window[1]+1] @ J_hom_init_t              # [89,4,31]

            # # back to [89,31,4]
            # J_tr_init_hom = J_tr_hom_init_t.transpose(1,2)   # [89,31,4]

            # # drop the homogeneous 1s → [89,31,3]
            # J_local_init = J_tr_init_hom[..., :3]

            # local_pose_diff = gmof(J_local - J_local_init).mean()


            # rotation = Ts[:,:3,:3]      
            # translation = Ts[:,:3,3]

            # full_joints2d = full_perspective_projection(
            #     joints3d,
            #     cam_intrinsics=self.cam_intrinsics,
            #     rotation=rotation,
            #     translation=translation,
            # )

            # scale = bbox_window[..., 2:].unsqueeze(-1) * 200.

            # # Loss 1. Data term
            # pred_keypoints = full_joints2d[..., :17, :]
            # joints_conf = input_keypoints_window[..., -1:]
            # reprojection_error = gmof(pred_keypoints - input_keypoints_window[..., :-1])
            # reprojection_error = ((reprojection_error * joints_conf) / scale).mean()


            loss_dict = {}
            # Compute the epipolar error
            epipolar_loss = compute_epipolar_error_relative(opt_rel, cam_intrinsics.squeeze(0), kp0_i, kp1_i)
            loss_dict['epipolar_loss'] = epipolar_loss
            # loss_dict['local_pose_diff'] = local_pose_diff*1000
            # loss_dict['reprojection'] = reprojection_error*1000
            
            loss = sum(loss_dict.values())
            loss.backward()

            # trans_window.grad.mul_(1000.0)

            return loss
        
        return closure
