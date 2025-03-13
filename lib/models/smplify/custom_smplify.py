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

        # transl_world = init_pred['trans_world'].squeeze(0)
        # poses_root_world = init_pred['poses_root_world'].squeeze(0).unsqueeze(1)
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
    # # Debugging
    # gt = joblib.load("/mnt/hdd/emdb_dataset/P4/36_outdoor_long_walk/P4_36_outdoor_long_walk_data.pkl")

    # gt_smpl = gt['smpl']

    # gt_pose = gt_smpl['poses_body']
    # gt_betas = gt_smpl['betas']
    # gt_t = gt['smpl']['trans']
    # gt_poses_root = gt_smpl['poses_root']

    # gt_pose = torch.tensor(gt_pose).float().to(device)
    # gt_poses_root = torch.tensor(gt_poses_root).float().to(device)

    # gt_pose = axis_angle_to_matrix(gt_pose.reshape(-1,23,3)).reshape(-1, 23, 3, 3)
    # gt_poses_root = axis_angle_to_matrix(gt_poses_root).reshape(-1,1, 3, 3)

    # output = smpl.get_output(betas=torch.from_numpy(gt_betas).to(device).view(-1,10).repeat(1,1), 
    #                             body_pose=gt_pose, 
    #                             global_orient=gt_poses_root, 
    #                             transl=torch.from_numpy(gt_t).to(device),
    #                             pose2rot=False,
    #                             return_full_pose=False)
    # gt_joints = output.joints.cpu()

    """"    
    gt_smpl = d['smpl']

    gt_pose = gt_smpl['poses_body'][:length,:]
    gt_betas = gt_smpl['betas']

    gt_poses_root = gt_smpl['poses_root'][:length,:]

    gt_pose = torch.tensor(gt_pose).float().to(self.device)
    gt_poses_root = torch.tensor(gt_poses_root).float().to(self.device)

    gt_pose = axis_angle_to_matrix(gt_pose.reshape(-1,23,3)).reshape(-1, 23, 3, 3)
    gt_poses_root = axis_angle_to_matrix(gt_poses_root).reshape(-1,1, 3, 3)

    output = self.smpl.get_output(betas=torch.from_numpy(gt_betas).to(self.device).view(-1,10).repeat(length,1), 
                                  body_pose=gt_pose, 
                                  global_orient=gt_poses_root, 
                                  transl=torch.from_numpy(gt_trans).to(self.device),
                                  pose2rot=False,
                                  return_full_pose=False)
    gt_joints = output.joints.cpu()

  

    transl_wham_world = transl_wham_world.cpu()
    transl_cam = transl_cam.cpu()
    transl_world = transl_world.cpu()
    t_cam_pose = t_cam_pose.cpu()
    gt_trans = gt_trans[:length, :3]
    wham_joints_world = torch.einsum("tij,tnj->tni", R_cam_pose, wham_joints_cam.to(self.device)) + t_cam_pose[:, None].to(self.device)
    wham_joints_world = wham_joints_world.cpu()
    num = 0
    cam_trans = np.linalg.inv(gt_extrinsics.squeeze().cpu())[:, :3, 3]
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
    
    # init_pred['trans_world'] = transl_world
    # init_pred['poses_root_world'] = poses_root_world

    # print('Optimizing global translation progressively...')
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    gt_trans = gt['smpl']['trans']
    trans_world_raw = init_pred['trans_world'].cpu()
    ax.scatter(trans_world_raw[:,0], trans_world_raw[:,1],trans_world_raw[:,2], c='g', marker='o', label='Aligned')
    ax.scatter(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], c='r', marker='o', label='GT')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    ax.set_title("Global Translation: GT vs Aligned")
    """
    # plt.show()


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
        # reprojected_points = full_perspective_projection(joints3d_world[0], cam_intrinsics=)

        # exit()


    opt_SMPL_params = custom_smplify.fit_SMPL_params(current_pred, transl_world_aligned, poses_root_world_aligned)

    print('Optimization complete.')


    """    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gt_trans = gt['smpl']['trans']
        trans_world = opt_SMPL_params['trans_world'].cpu()
        ax.scatter(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], c='r', marker='o', label='GT')
        ax.scatter(trans_world[:, 0], trans_world[:, 1], trans_world[:, 2], c='b', marker='o', label='Optimized')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title("Global Translation: GT vs Optimized vs Aligned")
        ax.legend()
    """    






    return opt_SMPL_params

def WA_MPJPE_align(cam, bbox, res, cam_intrinsics, smpl, device, 
                   pose, betas, transl_wham_world, poses_root_wham_world, gt_extrinsics):
    """
    Compute per-frame alignments from WHAM to camera and from camera to GT world,
    then average these to obtain a single transformation (R and t) that can be applied
    to the averaged WHAM translation and root orientation.
    
    Args:
        cam:                  Tensor of camera parameters, shape [B, n, ...]
        bbox:                 Tensor of bounding boxes, shape [B, n, ...]
        res:                  Tuple (width, height)
        cam_intrinsics:       Tensor of camera intrinsics, shape [B, n, 3, 3] (or similar)
        smpl:                 SMPL model with a forward_align() method.
        device:               Torch device.
        pose:                 SMPL pose parameters, shape [B, n, ...]
        betas:                SMPL shape parameters, shape [B, n, 10] or broadcastable.
        transl_wham_world:    Initial WHAM translation, shape [n, 3]
        poses_root_wham_world:Initial WHAM global orientation, shape [n, 3] (or a vector representation)
        gt_extrinsics:        GT camera extrinsics, shape [B, n, 4, 4]
        
    Returns:
        transl_world:         Final (averaged) translation in GT world coordinates, shape [3]
        poses_root_world:     Final (averaged) global orientation, shape [3] (or the appropriate representation)
    """
    # Assumption: batch dimension B is 1.
    n = cam.shape[1]  # number of frames
    
    # 1. Compute camera translation for each frame (batch processing if possible).
    trans_cam = convert_pare_to_full_img_cam(
                    cam, 
                    bbox[:, :, 2] * 200., 
                    bbox[:, :, :2], 
                    res[0], 
                    res[1], 
                    focal_length=cam_intrinsics[:, :, 0, 0]
                )
    
    # 2. For each frame, compute SMPL joints in camera and WHAM coordinates.
    joints3d_cam_list = []
    joints3d_wham_list = []
    for i in range(n):
        # Select frame i (assumed batch dimension remains)
        trans_cam_i = trans_cam[:, i]  # shape: [B, ...]
        output_cam = smpl.forward_align(pose[:, i:i+1], betas[:, i:i+1], trans_opt=trans_cam_i)
        joints3d_cam_list.append(output_cam.joints.cpu())
        
        output_wham = smpl.forward_align(
            pose[:, i:i+1], betas[:, i:i+1], 
            trans_opt=transl_wham_world[i:i+1], 
            global_orient_opt=poses_root_wham_world[i:i+1]
        )
        joints3d_wham_list.append(output_wham.joints.cpu())
    
    # Stack resulting joints: each will have shape [n, num_joints, 3]
    joints3d_cam_all = torch.cat(joints3d_cam_list, dim=0)
    joints3d_wham_all = torch.cat(joints3d_wham_list, dim=0)
    
    # 3. Compute per-frame WHAM-to-CAM alignment (rotation and translation).
    R_list = []
    t_list = []
    for i in range(n):
        _, R_i, t_i = first_align_joints_return_R_t(
            joints3d_cam_all[i:i+1], 
            joints3d_wham_all[i:i+1]
        )
        R_list.append(R_i.to(device))
        t_list.append(t_i.to(device))
    
    # Average the per-frame translations.
    avg_t_wham_cam = torch.mean(torch.stack(t_list, dim=0), dim=0)  # shape: [1,3] or [3]
    
    # Naively average rotations elementwise then project back to SO(3) via SVD.
    R_stack = torch.stack(R_list, dim=0).squeeze(1)  # shape: [n, 3, 3]
    avg_R_naive = torch.mean(R_stack, dim=0)  # shape: [3, 3]
    U, S, V = torch.svd(avg_R_naive)
    avg_R_wham_cam = U @ V.t()  # Averaged rotation from WHAM to CAM.
    
    # 4. For each frame, get the camera-to-GT world transformation.
    R_cam_list = []
    t_cam_list = []
    for i in range(n):
        # Assume gt_extrinsics is [B, n, 4, 4]; here B is 1.
        extr = gt_extrinsics[:, i, :, :].squeeze(0)  # shape: [4, 4]
        extr_inv = torch.inverse(extr)
        R_cam_i = extr_inv[:3, :3]
        t_cam_i = extr_inv[:3, 3]
        R_cam_list.append(R_cam_i.to(device))
        t_cam_list.append(t_cam_i.to(device))
    
    # Average the camera-to-world transformations.
    avg_t_cam = torch.mean(torch.stack(t_cam_list, dim=0), dim=0)  # shape: [3]
    R_cam_stack = torch.stack(R_cam_list, dim=0)  # shape: [n, 3, 3]
    avg_R_cam_naive = torch.mean(R_cam_stack, dim=0)
    U_cam, S_cam, V_cam = torch.svd(avg_R_cam_naive)
    avg_R_cam = U_cam @ V_cam.t()  # Averaged rotation from CAM to GT world.
        
    # 5. Compose the final averaged transformation:
    # Final rotation: from WHAM to GT = (CAM->GT) * (WHAM->CAM)
    R_final = avg_R_cam @ avg_R_wham_cam
    # Final translation: t_final = (CAM->GT)*avg_t_wham_cam + avg_t_cam
    t_final = (avg_R_cam @ avg_t_wham_cam.T).T + avg_t_cam

    # 6. Apply the final transformation to every frame of the original WHAM translation.
    #    transl_wham_world has shape [n, 3]. We want:
    #       transl_world[i] = R_final @ transl_wham_world[i] + t_final
    transl_world = (R_final @ transl_wham_world.T).T + t_final  # shape: [n, 3]

    # 7. Optionally, you might also transform the per-frame global orientation.
    # Here we apply the rotation to each frame's poses_root_wham_world.
    poses_root_world = R_final @ poses_root_wham_world.squeeze(1)  # shape: [n, 3]
    
    return transl_world, poses_root_world



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








        # gt_smpl = gt['smpl']

        # gt_pose = gt_smpl['poses_body'][window:window+1,:]
        # gt_betas = gt_smpl['betas']
        # gt_t = gt['smpl']['trans'][window:window+1,:]
        # gt_poses_root = gt_smpl['poses_root'][window:window+1,:]

        # gt_pose = torch.tensor(gt_pose).float().to(device)
        # gt_poses_root = torch.tensor(gt_poses_root).float().to(device)

        # gt_pose = axis_angle_to_matrix(gt_pose.reshape(-1,23,3)).reshape(-1, 23, 3, 3)
        # gt_poses_root = axis_angle_to_matrix(gt_poses_root).reshape(-1,1, 3, 3)

        # output = smpl.get_output(betas=torch.from_numpy(gt_betas).to(device).view(-1,10).repeat(1,1), 
        #                             body_pose=gt_pose, 
        #                             global_orient=gt_poses_root, 
        #                             transl=torch.from_numpy(gt_t).to(device),
        #                             pose2rot=False,
        #                             return_full_pose=False)
        # gt_joints = output.joints.cpu()
        # gt_trans = gt['smpl']['trans'][window:window+1,:]
        # print("done")
        # t_cam_pose = t_cam_pose.cpu()
        # wham_joints_world = torch.einsum("tij,tnj->tni", R_cam_pose, wham_joints_cam.to(device)) + t_cam_pose[:, None].to(device)
        # wham_joints_world = wham_joints_world.cpu()



        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # ax1 = fig.add_subplot(111, projection='3d')
        # trans_world_seq = sequence_transl_world.cpu()
        # ax.scatter(trans_world_seq[:,0], trans_world_seq[:,1],trans_world_seq[:,2], c='g', marker='o', label='Aligned')
        # ax.scatter(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], c='r', marker='o', label='GT')
        # ax.scatter(transl_world.cpu()[:,0] ,transl_world.cpu()[:,1], transl_world.cpu()[:,2], c='b', marker='o', label='Full')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.legend()
        # ax.set_title("Aligned vs GT")
        
        # ax1.scatter(joints3d_cam[:,:,0], joints3d_cam[:,:,1], joints3d_cam[:,:,2], c='r', marker='o', label='Cam')
        # ax1.scatter(wham_joints_cam[:,:,0], wham_joints_cam[:,:,1], wham_joints_cam[:,:,2], c='b', marker='o', label='WHAM in Cam')
        # ax1.scatter(wham_joints_world[:,:,0], wham_joints_world[:,:,1], wham_joints_world[:,:,2], c='m', marker='o', label='WHAM')
        # ax1.scatter(gt_joints[:,:,0], gt_joints[:,:,1], gt_joints[:,:,2], c='g', marker='o', label='GT')
        # ax1.set_xlabel('X Label')
        # ax1.set_ylabel('Y Label')
        # ax1.set_zlabel('Z Label')
        # ax1.legend()
        # print("Aligned")
        # plt.show()


    """     
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], c='r', marker='o', label='GT')
    ax.scatter(gt_joints[:,:,0], gt_joints[:,:,1], gt_joints[:,:,2], c='g', marker='o', label='GT')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    ax.set_title("GT joints and trans")
    print("Aligned")
    plt.show()
    """

    return transl_world, poses_root_world

def plot_3d_joints(joints, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


"""
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(gt_joints[:,:,0], gt_joints[:,:,1], gt_joints[:,:,2], c='g', marker='o', label='GT')
ax1.scatter(gt_trans[window,0], gt_trans[window,1], gt_trans[window,2], c='r', label='GT Trans')
"""