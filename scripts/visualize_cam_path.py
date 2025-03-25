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


    sequence_root_wham = get_sequence_root(args, gt=False)

    dpvo_path = glob(os.path.join(sequence_root_wham, "slam_results_gt_intrinsics.pth"))[0]
    dpvo_output = joblib.load(dpvo_path)
    print(len(dpvo_output))
  
    dpvo_orientation = R.from_quat(dpvo_output[:,3:]).as_matrix()
    dpvo_trans = dpvo_output[:,:3]
    # create dpvo_cam object

    # Create 4x4 transformation matrices for dpvo_cam
    dpvo_cam = np.eye(4)[None].repeat(len(dpvo_orientation), axis=0)
    dpvo_cam[:, :3, :3] = dpvo_orientation
    dpvo_cam[:, :3, 3] = dpvo_trans
    dpvo_extrinsics = invert_camera_poses(dpvo_cam)

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
    
    scale_dpvo = dpvo_extrinsics @ extrinsics[0]
    scale_pos = get_camera_position(scale_dpvo)

    path = glob(os.path.join(sequence_root_wham, "eval.pkl"))[0]
    output = joblib.load(path)
    smpl_trans = output["trans_world"]
    # smpl_trans = data['smpl']['trans']
    
    scale_pos = torch.tensor(scale_pos, dtype=torch.float32)
    smpl_trans = torch.tensor(data['smpl']['trans'], dtype=torch.float32)
    
    scale, _, _ = align_pcl(smpl_trans.unsqueeze(0), scale_pos.unsqueeze(0))

    dpvo_extrinsics[:, :3, 3] *= float(scale)

    dpvo_extrinsics = dpvo_extrinsics @ extrinsics[0]

    print(scale)

    gt_camera = OpenCVCamera(intrinsics, extrinsics[:, :3], cols, rows, viewer=viewer, name="GT Camera")
    dpvo_camera = OpenCVCamera(intrinsics, dpvo_extrinsics[:,:3], cols, rows, viewer=viewer, name="DPVO Camera")

    # Display the images on a billboard.
    raw_images_bb = Billboard.from_camera_and_distance(
        gt_camera,
        10.0,
        cols,
        rows,
        image_files,
        image_process_fn=drawing_function(kp2d, bboxes),
        name="Image",
    )

    viewer.scene.add(gt_camera, dpvo_camera)


    if args.draw_trajectories:
        # Add a path trail for the SMPL root trajectory.
        
        cam_pos = get_camera_position(extrinsics)
        gt_camera_path = LinesTrail(
            cam_pos,
            r_base=0.03,
            color=(0.8, 0.2, 0.2, 1),
            cast_shadow=False,
            name="Camera Trajectory: GT",
        )

        dpvo_pos = get_camera_position(dpvo_extrinsics)
        dpvo_cam_path = LinesTrail(
            dpvo_pos,
            r_base=0.03,
            color=(0.2, 0.8, 0.2, 1),
            cast_shadow=False,
            name="Camera Trajectory: DPVO",
        )

        if not args.mini:
            viewer.scene.add(gt_camera_path, dpvo_cam_path)
        else:
            viewer.scene.add(gt_camera_path)

    # Remaining viewer setup.
    if args.view_from_camera:
        # We view the scene through the camera.
        viewer.set_temp_camera(dpvo_camera)
    else:
        # We center the scene on the first frame of the SMPL sequence.
        pass

    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = False
    viewer.playback_fps = 30.0

    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--subject", type=str, default=_C.subject_id, help="The subject ID, P0 - P9.")
    parser.add_argument(
        "--sequence",
        type=str,
        default=_C.sequence_id,
        help="The sequence ID. This can be any unambiguous prefix of the sequence's name, i.e. for the "
        "sequence '66_outdoor_rom' it could be '66' or any longer prefix including the full name.",
    )
    parser.add_argument(
        "--view_from_camera",
        action="store_true",
        help="View it from the camera's perspective.",
        default=True
    )
    parser.add_argument(
        "--draw_2d",
        action="store_true",
        help="Draw 2D keypoints and bounding boxes on the image.",
    )
    parser.add_argument(
        "--draw_trajectories",
        action="store_true",
        help="Render SMPL and camera trajectories.",
        default=True
    )
    parser.add_argument(
        "--mini",
        action='store_true',
        default=False
    )

    parser.add_argument("--gt_camera", default=True, action='store_true', help="Use ground truth camera pose")


    args = parser.parse_args()

    C.update_conf({"smplx_models": SMPLX_MODELS})

    main(args)
