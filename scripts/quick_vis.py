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


import sys

sys.path.append("/home/felix/WHAM_MA")

from lib.eval.eval_utils import first_align_joints_return_R_t

from lib.utils.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R
import torch
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
        WHAM_OUTPUT = _C.PATHS.WHAM_OUTPUT
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


    wham_gt_intrinsics_iference_data_path = glob(os.path.join(sequence_root_wham, "*_output_DPVO.pkl"))[0]
    wham_output = joblib.load(wham_gt_intrinsics_iference_data_path)



    inf_smpl_layer = SMPLLayer(model_type="smpl", gender="male")

    wham_smpl_seq = SMPLSequence(
        wham_output["pose_world"][:,3:],
        smpl_layer=inf_smpl_layer,
        poses_root=wham_output["pose_world"][:,:3],
        betas=wham_output["betas"],
        # trans=wham_output["trans_world"],
        trans=wham_output["trans_world"],
        # trans=data["smpl"]["trans"],
        name="Mesh: WHAM",
        color = (0.2, 0.8, 0.2, 1),
    )    


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

    viewer = Viewer()

    viewer.scene.add(wham_smpl_seq, gt_smpl_seq)
    viewer.scene.add(wham_path, gt_path)
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
