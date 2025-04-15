import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

from lib.utils import transforms

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.data.dataloader import setup_eval_dataloader
from lib.utils.utils import prepare_batch

from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.models.smplify.custom_smplify import CustomSMPLify
from lib.data.datasets.dataset_custom import convert_dpvo_to_cam_angvel

from scripts.custom_utils import get_sequence_root, find_substring
from configs import constants as _C

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from lib.models.smplify.custom_smplify import optimization_upper_bound, optimization_baseline, W_MPJPE_align

from smplx import SMPL

from configs.config import get_cfg_defaults
from configs.config import parse_args

from scripts.align_emdb import align_and_compute_metrics
from scripts.visualize_cam_path import invert_camera_poses


def run(args,
        output_pth,
        ):

    sequence_root = get_sequence_root(args)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]
 
    if args.upper_bound:
        pth = osp.join(output_pth, "upper_bound_gt_betas.pkl")
    elif args.baseline:
        pth = osp.join(output_pth, "baseline_gt_betas.pkl")
    else:
        pth = osp.join(output_pth, "wham_raw_output_gt_betas.pkl")

    align_and_compute_metrics(gt_data_path, pth, cfg)

if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    # # Output folder
    sequence = args.subject + "_" + args.sequence
    # print(sequence)
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(args, 
        output_pth
    )
        
    logger.info('Done !')

    # run(args,
    #     output_pth)