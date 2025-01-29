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
        print(os.path.join(EMDB_ROOT,args.subject, sequence_id + "*"))
        candidates = glob(os.path.join(EMDB_ROOT,args.subject, sequence_id + "*"))
        print(candidates)
        if len(candidates) == 0:
            raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
        elif len(candidates) > 1:
            raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
        return candidates[0]


    else:
        WHAM_OUTPUT = "/home/felix/WHAM_MA/output/emdb"
        """Parse the path of the sequence to be visualized."""
        print
        sequence_id = "{:0>2d}".format(int(args.sequence))
        print(os.path.join(WHAM_OUTPUT,args.subject+"_"+sequence_id))
        candidates = glob(os.path.join(WHAM_OUTPUT,args.subject+"_"+sequence_id))
        print(candidates)
        if len(candidates) == 0:
            raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
        elif len(candidates) > 1:
            raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
        return candidates[0]




def main(args):
    # Access EMDB data.
    sequence_root = get_sequence_root(args)
    data_file = glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    print(data_file)
    with open(data_file, "rb") as f:
        data = pkl.load(f)

    print(data["good_frames_mask"].sum())

    # Set up SMPL layer (RED).
    gender = data["gender"]
    gt_smpl_layer = SMPLLayer(model_type="smpl", gender=gender)

    gt_smpl_seq = SMPLSequence(
        data["smpl"]["poses_body"],
        smpl_layer=gt_smpl_layer,
        poses_root=data["smpl"]["poses_root"],
        betas=data["smpl"]["betas"].reshape((1, -1)),
        trans=data["smpl"]["trans"],
        name="Mesh: GT",
        color=(0.8, 0.2, 0.2, 1),
    )


    # (GREEN)
    sequence_root_wham = get_sequence_root(args, gt=False)
    wham_iference_data_path = glob(os.path.join(sequence_root_wham, "*_output_DPVO_processed.pkl"))[0]
    wham_output = joblib.load(wham_iference_data_path)
    wham_output = wham_output[0]
    inf_smpl_layer = SMPLLayer(model_type="smpl", gender=gender)

    wham_smpl_seq = SMPLSequence(
        wham_output["pose_world"][:,3:],
        smpl_layer=inf_smpl_layer,
        poses_root=wham_output["pose_world_hat"][:,:3],
        betas=wham_output["betas"],
        # trans=wham_output["trans_world"],
        trans=wham_output["trans_world_hat"],
        # trans=data["smpl"]["trans"],
        name="Mesh: WHAM",
        color = (0.2, 0.8, 0.2, 1),
    )    

    wham_gt_cam_iference_data_path = glob(os.path.join(sequence_root_wham, "*_output_gt_camera_processed.pkl"))[0]
    wham_gt_cam_output = joblib.load(wham_gt_cam_iference_data_path)
    wham_gt_cam_output = wham_gt_cam_output[0]
    wham_gt_cam_smpl_layer = SMPLLayer(model_type="smpl", gender=gender)
    wham_gt_cam_smpl_seq = SMPLSequence(
        wham_gt_cam_output["pose_world"][:,3:],
        smpl_layer=wham_gt_cam_smpl_layer,
        poses_root=wham_gt_cam_output["pose_world_hat"][:,:3],
        betas=wham_gt_cam_output["betas"],
        # trans=wham_output["trans_world"],
        trans=wham_gt_cam_output["trans_world_hat"],
        # trans=data["smpl"]["trans"],
        name="Mesh: WHAM + GT Cam",
        color = (0.2, 0.2, 0.8, 1),
    )  

    wham_gt_intrinsics_iference_data_path = glob(os.path.join(sequence_root_wham, "*_output_gt_intrinsics_processed.pkl"))[0]
    wham_gt_intrinsics_output = joblib.load(wham_gt_intrinsics_iference_data_path)
    wham_gt_intrinsics_output = wham_gt_intrinsics_output[0]
    wham_gt_intrinsics_smpl_layer = SMPLLayer(model_type="smpl", gender=gender)
    wham_gt_intrinsics_smpl_seq = SMPLSequence(
        wham_gt_intrinsics_output["pose_world"][:,3:],
        smpl_layer=wham_gt_intrinsics_smpl_layer,
        poses_root=wham_gt_intrinsics_output["pose_world_hat"][:,:3],
        betas=wham_gt_intrinsics_output["betas"],
        # trans=wham_output["trans_world"],
        trans=wham_gt_intrinsics_output["trans_world_hat"],
        # trans=data["smpl"]["trans"],
        name="Mesh: WHAM + GT Intrinsics",
        color = (0.2, 0.8, 0.8, 1),
    )    

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

    wham_extrinsics = wham_output["cam_pose_hat"]
    wham_extrinsics = np.linalg.inv(wham_extrinsics)

    wham_extrinsics_gt_intrinsics = wham_gt_intrinsics_output["cam_pose_hat"]
    wham_extrinsics_gt_intrinsics = np.linalg.inv(wham_extrinsics_gt_intrinsics)

    # Create the viewer
    viewer_size = None
    if args.view_from_camera:
        target_height = 1080
        width = int(target_height * cols / rows)
        viewer_size = (width, target_height)

        # If we view it from the camera drawing the 3D trajectories might be disturbing, suppress it.
        args.draw_trajectories = False

    viewer = Viewer(size=viewer_size)

    # Prepare the camera.
    intrinsics = np.repeat(intrinsics[np.newaxis, :, :], len(extrinsics), axis=0)

    gt_camera = OpenCVCamera(intrinsics, extrinsics[:, :3], cols, rows, viewer=viewer, name="GT Camera")

    wham_camera = OpenCVCamera(intrinsics, wham_extrinsics[:, :3], cols, rows, viewer=viewer, name="DPVO Camera")

    wham_camera_gt_intrinsics = OpenCVCamera(intrinsics, wham_extrinsics_gt_intrinsics[:, :3], cols, rows, viewer=viewer, name="DPVO GT Intrinsics Camera")
    
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

    # Add everything to the scene.

    if args.gt_camera:
        viewer.scene.add(raw_images_bb, gt_camera, wham_camera, wham_camera_gt_intrinsics, gt_smpl_seq, wham_smpl_seq, wham_gt_intrinsics_smpl_seq, wham_gt_cam_smpl_seq)
    else:
        viewer.scene.add(gt_camera, wham_camera, gt_smpl_seq, wham_smpl_seq, raw_images_bb)

    if args.draw_trajectories:
        # Add a path trail for the SMPL root trajectory.
        gt_path = LinesTrail(
            gt_smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0.8, 0.2, 0.2, 0.8),
            cast_shadow=False,
            name="SMPL Trajectory: GT",
        )

        wham_path = LinesTrail(
            wham_smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0.2, 0.8, 0.2, 0.8),
            cast_shadow=False,
            name="SMPL Trajectory: WHAM",
        )

        wham_gt_intrinsics_path = LinesTrail(
            wham_gt_intrinsics_smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0.2, 0.2, 0.8, 0.8),
            cast_shadow=False,
            name="SMPL Trajectory: WHAM + GT Intrinsics",
        )

        wham_gt_cam_path = LinesTrail(
            wham_gt_cam_smpl_seq.joints[:, 0],
            r_base=0.003,
            color=(0.2, 0.8, 0.8, 0.8),
            cast_shadow=False,
            name="SMPL Trajectory: WHAM + GT Cam",
        )

        # Add a path trail for the camera trajectory.
        # A fixed path (i.e. not a trail), could also be enabled in the GUI on the camera node by clicking "Show path".
        cam_pos = get_camera_position(extrinsics)
        gt_camera_path = LinesTrail(
            cam_pos,
            r_base=0.003,
            color=(0.5, 0.5, 0.5, 1),
            cast_shadow=False,
            name="Camera Trajectory: GT",
        )

        wham_cam_pos = get_camera_position(wham_extrinsics)
        dpvo_path = LinesTrail(
            wham_cam_pos,
            r_base=0.003,
            color=(0.5, 0.8, 0.8, 1),
            cast_shadow=False,
            name="Camera Trajectory: DPVO",
        )

        wham_gt_intrinsics_cam_pos = get_camera_position(wham_extrinsics_gt_intrinsics)
        dpvo_gt_intrinsics_cam_path = LinesTrail(
            wham_gt_intrinsics_cam_pos,
            r_base=0.003,
            color=(0.5, 0.8, 0.8, 1),
            cast_shadow=False,
            name="Camera Trajectory: DPVO + GT Intrinsics",
        )

        if args.gt_camera:
            viewer.scene.add(gt_camera_path, dpvo_path, dpvo_gt_intrinsics_cam_path, gt_path, wham_path, wham_gt_intrinsics_path, wham_gt_cam_path)
        else:
            viewer.scene.add(gt_camera_path, dpvo_path, gt_path, wham_path)

    # Remaining viewer setup.
    if args.view_from_camera:
        # We view the scene through the camera.
        viewer.set_temp_camera(gt_camera)

        # Hide all the GUI controls, they can be re-enabled by pressing `ESC`.
        viewer.render_gui = False
    else:
        # We center the scene on the first frame of the SMPL sequence.
        viewer.center_view_on_node(gt_smpl_seq)

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

    parser.add_argument("--gt_camera", default=True, action='store_true', help="Use ground truth camera pose")


    args = parser.parse_args()

    C.update_conf({"smplx_models": SMPLX_MODELS})

    main(args)
