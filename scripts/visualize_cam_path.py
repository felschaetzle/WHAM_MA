"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann

Script to visualize an EMDB sequence. Make sure to set the path of `EMDB_ROOT` and `SMPLX_MODELS` below.

Usage:
  python visualize.py P8 68_outdoor_handstand
"""
import argparse
from glob import glob
import os
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.lines import LinesTrail
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer

import torch
from scipy.spatial.transform import Rotation as R

import joblib

import sys
sys.path.append("/home/felix/WHAM_MA")
from configs import constants as _C

from emdb_configuration import (
    EMDB_ROOT,
    SMPL_SIDE_COLOR,
    SMPL_SIDE_INDEX,
    SMPL_SKELETON,
    SMPLX_MODELS,
)

from lib.eval.eval_utils import align_pcl
from lib.utils.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

from configs.config import parse_args

def draw_kp2d(kp2d, bboxes=None):
    """Draw 2D keypoints and bounding boxes on the image with OpenCV."""

    def _draw_kp2d(img, current_frame_id):
        current_kp2d = kp2d[current_frame_id].copy()
        scale = img.shape[0] / 1000

        # Draw lines.
        for index in range(SMPL_SKELETON.shape[0]):
            i, j = SMPL_SKELETON[index]
            # color = SIDE_COLOR[max(SIDE_INDEX[i], SIDE_INDEX[j])]
            cv2.line(
                img,
                tuple(current_kp2d[i, :2].astype(np.int32)),
                tuple(current_kp2d[j, :2].astype(np.int32)),
                (0, 0, 0),
                int(scale * 3),
            )

        # Draw points.
        for jth in range(0, kp2d.shape[1]):
            color = SMPL_SIDE_COLOR[SMPL_SIDE_INDEX[jth]]
            radius = scale * 5

            out_color = (0, 0, 0)
            in_color = color

            img = cv2.circle(
                img,
                tuple(current_kp2d[jth, :2].astype(np.int32)),
                int(radius * 1.4),
                out_color,
                -1,
            )
            img = cv2.circle(
                img,
                tuple(current_kp2d[jth, :2].astype(np.int32)),
                int(radius),
                in_color,
                -1,
            )

        # Draw bounding box if available.
        if bboxes is not None:
            bbox = bboxes[current_frame_id]
            x_min, y_min, x_max, y_max = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        return img

    return _draw_kp2d


def draw_nothing(kp2d, bboxes=None):
    """Dummy function."""

    def _draw_nothing(img, current_frame_id):
        return img

    return _draw_nothing


def get_camera_position(Rt):
    """Get the orientation and position of the camera in world space."""
    pos = -np.transpose(Rt[:, :3, :3], axes=(0, 2, 1)) @ Rt[:, :3, 3:]
    return pos.squeeze(-1)


def get_sequence_root(args, gt=True):
    if gt:
        """Parse the path of the sequence to be visualized."""
        sequence_id = "{:0>2d}".format(int(args.sequence))
        candidates = glob(os.path.join(EMDB_ROOT,args.subject, sequence_id + "*"))
        if len(candidates) == 0:
            raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
        elif len(candidates) > 1:
            raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
        return candidates[0]


    else:
        WHAM_OUTPUT = _C.PATHS.WHAM_OUTPUT
        """Parse the path of the sequence to be visualized."""
        sequence_id = "{:0>2d}".format(int(args.sequence))
        candidates = glob(os.path.join(WHAM_OUTPUT,args.subject+"_"+sequence_id))
        if len(candidates) == 0:
            raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
        elif len(candidates) > 1:
            raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
        return candidates[0]

def invert_camera_poses(pose):
    """Invert camera-to-world pose matrices into extrinsics (world-to-camera)."""
    R_c2w = pose[:, :3, :3]
    t_c2w = pose[:, :3, 3:]
    
    R_w2c = np.transpose(R_c2w, axes=(0, 2, 1))
    t_w2c = - R_w2c @ t_c2w
    
    extrinsics = np.eye(4)[None].repeat(len(pose), axis=0)
    extrinsics[:, :3, :3] = R_w2c
    extrinsics[:, :3, 3:] = t_w2c
    return extrinsics

def main(args):
    # Access EMDB data.
    sequence_root = get_sequence_root(args)
    data_file = glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    print(data_file)
    with open(data_file, "rb") as f:
        data = pkl.load(f)

    # Set up SMPL layer (RED).
    gender = data["gender"]
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender)

    gt_smpl_seq = SMPLSequence(
        data["smpl"]["poses_body"],
        smpl_layer=smpl_layer,
        poses_root=data["smpl"]["poses_root"],
        betas=data["smpl"]["betas"].reshape((1, -1)),
        trans=data["smpl"]["trans"],
        name="Mesh: GT",
        color=(0.8, 0.2, 0.2, 1),
    )

    if args.baseline:
        method = "baseline_gt_betas.pkl"
        print("Using baseline")
    if args.upper_bound:
        method = "upper_bound_gt_betas.pkl"
        print("Using upper bound")
    
    if args.baseline and args.upper_bound or not args.baseline and not args.upper_bound:
        raise ValueError("Please specify either baseline or upper_bound")
    

    sequence_root_wham = get_sequence_root(args, gt=False)
    path = glob(os.path.join(sequence_root_wham, "baseline.pkl"))[0]
    output = joblib.load(path)

    # seq = SMPLSequence(
    #     output["pose_world"][:,3:],
    #     smpl_layer=smpl_layer,
    #     poses_root=output["pose_world"][:,:3],
    #     betas=output["betas"],
    #     trans=output["trans_world"],
    #     name="Upper Bound/ Baseline",
    #     color = (0.8, 0.8, 0.2, 1),
    # )  

    # align_seq = SMPLSequence(
    #     output["pose_world"][:,3:],
    #     smpl_layer=smpl_layer,
    #     poses_root=output["pose_world_align"][:,:3],
    #     betas=output["betas"],
    #     trans=output["trans_world_align"],
    #     name="WHAM align",
    #     color = (0.2, 0.8, 0.8, 1),
    # )    

    path = glob(os.path.join(sequence_root_wham, method))[0]
    output = joblib.load(path)

    beta_seq = SMPLSequence(
        output["pose_world"][:,3:],
        smpl_layer=smpl_layer,
        poses_root=output["pose_world"][:,:3],
        betas=output["betas"],
        trans=output["trans_world"],
        name="Upper Bound/ Baseline GT Beta",
        color = (0.8, 0.5, 0.2, 1),
    )  

    align_beta_seq = SMPLSequence(
        output["pose_world"][:,3:],
        smpl_layer=smpl_layer,
        poses_root=output["pose_world_align"][:,:3],
        betas=output["betas"],
        trans=output["trans_world_align"],
        name="WHAM align GT Beta",
        color = (0.2, 0.5, 0.8, 1),
    )    

    # dpvo_path = glob(os.path.join(sequence_root_wham, "slam_results_gt_intrinsics.pth"))[0]
    # dpvo_output = joblib.load(dpvo_path)
    # print(len(dpvo_output))
  
    # dpvo_orientation = R.from_quat(dpvo_output[:,3:]).as_matrix()
    # dpvo_trans = dpvo_output[:,:3]

    # # Create 4x4 transformation matrices for dpvo_cam
    # dpvo_cam = np.eye(4)[None].repeat(len(dpvo_orientation), axis=0)
    # dpvo_cam[:, :3, :3] = dpvo_orientation
    # dpvo_cam[:, :3, 3] = dpvo_trans

    # Create the viewer
    viewer_size = None

    viewer = Viewer(size=viewer_size)

    # Load 2D information.
    kp2d = data["kp2d"]
    bboxes = data["bboxes"]["bboxes"]
    drawing_function = draw_kp2d if args.draw_2d else draw_nothing

    # Load images.
    image_dir = os.path.join(sequence_root, "images")
    image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))

    # Load camera information.
    intrinsics = data["camera"]["intrinsics"]
    extrinsics = data["camera"]["extrinsics"]
    cols, rows = data["camera"]["width"], data["camera"]["height"]

    # Prepare the camera.
    intrinsics = np.repeat(intrinsics[np.newaxis, :, :], len(extrinsics), axis=0)
    

    if args.baseline:
        dpvo_extrinsics = output['dpvo_extrinsics_unscaled']

        aux_dpvo = dpvo_extrinsics.cpu() @ extrinsics[0]

        aux_dpvo_cam_pose = get_camera_position(aux_dpvo)


        scale, _, _ = align_pcl(torch.from_numpy(output['trans_world_align']).float().unsqueeze(0).cpu(), 
                                aux_dpvo_cam_pose[data['good_frames_mask']].unsqueeze(0).float())
        print(scale)
        dpvo_extrinsics[:, :3, 3] *= float(scale)
        dpvo_extrinsics = dpvo_extrinsics.cpu() @ extrinsics[0]

        dpvo_extrinsics = dpvo_extrinsics.cpu().numpy()
        # print("shape should be same", dpvo_extrinsics.shape)
        dpvo_extrinsics_ = output['dpvo_extrinsics']
        # print("shape should be same", dpvo_extrinsics.shape)
        print('Saity diff', (dpvo_extrinsics - dpvo_extrinsics_).mean())
        print('Scale diff', (scale - output['dpvo_scale']))

        dpvo_camera = OpenCVCamera(intrinsics, dpvo_extrinsics[:,:3], cols, rows, viewer=viewer, name="DPVO Camera")


        # Display the images on a billboard.
        dpvo_images_bb = Billboard.from_camera_and_distance(
            dpvo_camera,
            10.0,
            cols,
            rows,
            image_files,
            image_process_fn=drawing_function(kp2d, bboxes),
            name="Image DPVO",
        )

        viewer.scene.add(dpvo_images_bb, dpvo_camera)

    gt_camera = OpenCVCamera(intrinsics, extrinsics[:, :3], cols, rows, viewer=viewer, name="GT Camera")

    # Display the images on a billboard.
    raw_images_bb = Billboard.from_camera_and_distance(
        gt_camera,
        10.0,
        cols,
        rows,
        image_files,
        image_process_fn=drawing_function(kp2d, bboxes),
        name="Image GT",
    )




    if not args.mini:
        viewer.scene.add(raw_images_bb, gt_camera, gt_smpl_seq, beta_seq, align_beta_seq)
    else:
        pass
    if args.draw_trajectories:
        # Add a path trail for the SMPL root trajectory.
        
        gt_path = LinesTrail(
            gt_smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0.8, 0.2, 0.2, 0.8),
            cast_shadow=False,
            name="Trajectory: GT",
        )

        # path = LinesTrail(
        #     seq.joints[:, 0],
        #     r_base=0.003,
        #     color=(0.2, 0.8, 0.2, 0.8),
        #     cast_shadow=False,
        #     name="Trajectory: Basline/Upper bound",
        # )

        # align_path = LinesTrail(
        #     align_seq.joints[:, 0],
        #     r_base=0.003,
        #     color=(0.2, 0.8, 0.2, 0.8),
        #     cast_shadow=False,
        #     name="Trajectory: Basline/U. b. align only",
        # )


        beta_path = LinesTrail(
            beta_seq.joints[:, 0],
            r_base=0.003,
            color=(0.8, 0.8, 0.2, 0.8),
            cast_shadow=False,
            name="Trajectory: Basline/Upper bound GT Beta",
        )


        beta_align_path = LinesTrail(
            align_beta_seq.joints[:, 0],
            r_base=0.003,
            color=(0.8, 0.8, 0.2, 0.8),
            cast_shadow=False,
            name="Trajectory: Basline/U. b. align GT Beta",
        )

        cam_pos = get_camera_position(extrinsics)
        gt_camera_path = LinesTrail(
            cam_pos,
            r_base=0.003,
            color=(0.8, 0.2, 0.2, 1),
            cast_shadow=False,
            name="Camera Trajectory: GT",
        )

        if args.baseline:

            dpvo_pos = get_camera_position(dpvo_extrinsics)
            dpvo_cam_path = LinesTrail(
                dpvo_pos,
                r_base=0.003,
                color=(0.2, 0.8, 0.2, 1),
                cast_shadow=False,
                name="Camera Trajectory: DPVO",
            )
            viewer.scene.add(dpvo_cam_path)
        if not args.mini:
            viewer.scene.add(gt_camera_path, gt_path, beta_path, beta_align_path)
        else:
            pass
    # Remaining viewer setup.
    if args.view_from_camera:
        # We view the scene through the camera.
        if args.baseline:
            viewer.set_temp_camera(dpvo_camera)
        else:
            viewer.set_temp_camera(gt_camera)
    else:
        # We center the scene on the first frame of the SMPL sequence.
        pass

    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = False
    # viewer.scene.raw_images_bb.enabled = False
    viewer.playback_fps = 30.0

    viewer.run()


if __name__ == "__main__":
    cfg, cfg_files, args = parse_args(test=True)

    C.update_conf({"smplx_models": SMPLX_MODELS})

    main(args)
