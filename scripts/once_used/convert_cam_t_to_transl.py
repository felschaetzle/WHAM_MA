from lib.data.datasets import CustomDataset
from configs.config import get_cfg_defaults

import os
from glob import glob
import os.path as osp

import argparse
import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar
from configs import constants as _C
from scripts.custom_utils import get_sequence_root

from lib.models.smplify import TemporalSMPLify

def run(cfg,
        video,
        output_pth,
        calib=None):

    if calib is None:
        tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
        slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    else:
        tracking_results = joblib.load(osp.join(output_pth, 'tracking_results_gt_intrinsics.pth'))
        slam_results = joblib.load(osp.join(output_pth, 'slam_results_gt_intrinsics.pth'))
        logger.info(f'Already processed data exists at {output_pth} ! Load the data .')

    cap = cv2.VideoCapture(video)    
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)


    # load intrinsics
    if calib is not None:
            calib = np.loadtxt(calib, delimiter=" ")
            fx, fy, cx, cy = calib[:4]
            gt_intrinsics = np.eye(3)
            gt_intrinsics[0,0] = fx
            gt_intrinsics[0,2] = cx
            gt_intrinsics[1,1] = fy
            gt_intrinsics[1,2] = cy
            gt_intrinsics = torch.tensor(gt_intrinsics).float().to(cfg.DEVICE).unsqueeze(0).unsqueeze(0)
            kwargs['cam_intrinsics'] = gt_intrinsics


    smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE, gt_intrinsics=True)
    input_keypoints = dataset.tracking_results[_id]['keypoints']
    pred = smplify.fit(pred, input_keypoints, **kwargs)
    
    with torch.no_grad():
        network.pred_pose = pred['pose']
        network.pred_shape = pred['betas']
        network.pred_cam = pred['cam']
        output = network.forward_smpl(**kwargs)
        pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        





    cx, cy = datum["box_center"][person_id]
    bbox_size = datum["box_size"][person_id]
    img_size = datum["img_size"][person_id]
    W, H = img_size

    # for cam_t we use pred_cam rather than pred_cam_t
    cam_t = datum["pred_cam"][person_id]
    tz, tx, ty = cam_t
    scale = 2 / max(bbox_size * tz, 1e-9)
    tz = focal_length * scale
    tx = tx + scale * (cx - W * 0.5)
    ty = ty + scale * (cy - H * 0.5)
    cam_t = np.array([tx, ty, tz])


    s, tx, ty = pare_cam[..., 0], pare_cam[..., 1], pare_cam[..., 2]
    res = crop_res
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[..., 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[..., 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)



if __name__ == '__main__':
    subject_id = _C.subject_id
    sequence_id = _C.sequence_id

    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='/mnt/hdd/emdb_dataset/P5/40_indoor_walk_big_circle/raw.mov', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/smplify', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default="output/smplify/"+subject_id+"_"+sequence_id+"/gt_intrinsics.txt", 
                        help='Camera calibration file path')
    
    parser.add_argument("--gt_extrinsics", action='store_true', help="Use ground truth camera pose")

    parser.add_argument("--subject", type=str, default=subject_id, help="The subject ID, P0 - P9.")

    parser.add_argument("--sequence", type=str,default=sequence_id, help="The sequence ID, 66")



    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    
    sequence_root = get_sequence_root(args)
    video_path = glob(os.path.join(sequence_root, "*.mov"))[0]

    # Output folder
    sequence = args.subject + "_" + args.sequence
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        video_path, 
        output_pth, 
        args.calib)
        
    print()
    logger.info('Done !')