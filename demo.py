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
from lib.utils.transforms import matrix_to_axis_angle
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
from lib.models.smplify.custom_smplify import progressive_global_translation_optimization

from smplx import SMPL

from configs.config import get_cfg_defaults
from configs.config import parse_args

from scripts.align_emdb import align_and_compute_metrics

def run(cfg,
        args,
        video,
        output_pth,
        network,
        save_pkl):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    calib = "output/emdb/"+ args.subject + "_" + args.sequence + "/gt_intrinsics.txt"

    eval_loader = setup_eval_dataloader(cfg, 'emdb', args.eval_split, cfg.MODEL.BACKBONE)
    emdb_sequence_index = find_substring(args.subject+"_"+args.sequence, eval_loader.dataset.labels['vid'])
    if emdb_sequence_index is None:
        logger.error(f"Sequence {args.subject}_{args.sequence} not found in the emdb2 dataset. Not usefull for global trajectory.")
        return

    print("Loading data from eval loader")

    length_update = eval_loader.dataset.labels['frame_id'][emdb_sequence_index].shape[0] - 1
    print("Found # frames in dataset: ", length)
    print("Found # frames in eval loader: ", length_update)
    length = length_update

    calib_data = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib_data[:4]
    gt_intrinsics = np.eye(3)
    gt_intrinsics[0,0] = fx
    gt_intrinsics[0,2] = cx
    gt_intrinsics[1,1] = fy
    gt_intrinsics[1,2] = cy
    gt_intrinsics = torch.tensor(gt_intrinsics).float().to(cfg.DEVICE).unsqueeze(0)

    sequence_root = get_sequence_root(args)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    gt_data = joblib.load(gt_data_path)
    gt_extrinsics = gt_data["camera"]["extrinsics"]
    gt_cam_pose = np.linalg.inv(gt_extrinsics)
    gt_cam_pose_rot = R.from_matrix(gt_cam_pose[:,:3,:3]).as_quat()
    gt_cam_pose = np.concatenate([gt_cam_pose[:,:3,3], gt_cam_pose_rot], axis=1)

    results = defaultdict(dict)
    
    # WHAM uses Y-down coordinate system, while EMDB dataset uses Y-up one.
    yup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(cfg.DEVICE)

    with torch.no_grad():

        batch = eval_loader.dataset.load_data(emdb_sequence_index, flip=False)
        x, inits, features, kwargs, gt = prepare_batch(batch, cfg.DEVICE, cfg.TRAIN.STAGE == 'stage2')
        cam_angvel = kwargs['cam_angvel']

        cam2yup = batch['R'][0][:1].to(cfg.DEVICE)
        cam2ydown = cam2yup @ yup2ydown
        cam2root = transforms.rotation_6d_to_matrix(inits[1][:, 0, 0])
        ydown2root = cam2ydown.mT @ cam2root
        ydown2root = transforms.matrix_to_rotation_6d(ydown2root)
        kwargs['init_root'][:, 0] = ydown2root

        # Forward pass with flipped input
        flipped_batch = eval_loader.dataset.load_data(emdb_sequence_index, flip=True)
        f_x, f_inits, f_features, f_kwargs, f_gt = prepare_batch(flipped_batch, cfg.DEVICE, cfg.TRAIN.STAGE == 'stage2')


        if not args.naive_intrinsics:
            kwargs['cam_intrinsics'] = gt_intrinsics.unsqueeze(0)
            f_kwargs['cam_intrinsics'] = gt_intrinsics.unsqueeze(0)

        flipped_pred = network(f_x, f_inits, f_features, **f_kwargs)
        pred = network(x, inits, features, **kwargs)

        # Merge two predictions
        flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
        pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
        flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
        avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
        avg_pose = avg_pose.reshape(-1, 144)
        avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
        
        # Refine trajectory with merged prediction
        network.pred_pose = avg_pose.view_as(network.pred_pose)
        network.pred_shape = avg_shape.view_as(network.pred_shape)
        network.pred_contact = avg_contact.view_as(network.pred_contact)
        output = network.forward_smpl(**kwargs)
        pred = network.refine_trajectory(output, return_y_up=True, **kwargs)

    if args.run_baseline:
        if args.use_gt_betas:
            gt_betas = gt_data["smpl"]["betas"]
            gt_betas = gt_betas.reshape(1, 1, 10)
            gt_betas = np.repeat(gt_betas, repeats=length, axis=1)
            gt_betas = torch.tensor(gt_betas).float().to(cfg.DEVICE)
            pred['betas'] = gt_betas

        kwargs["gt_extrinsics"] = torch.tensor(gt_extrinsics).float().to(cfg.DEVICE).unsqueeze(0)
        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        pred = progressive_global_translation_optimization(
            pred, input_keypoints, kwargs['bbox'],
            kwargs['gt_extrinsics'], kwargs['cam_intrinsics'],
            smpl, cfg.DEVICE, length, kwargs['res'][0,:])

    if args.run_smplify:
        smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].cpu().numpy()
        pred = smplify.fit(pred, input_keypoints, **kwargs)
        
        with torch.no_grad():
            network.pred_pose = pred['pose']
            network.pred_shape = pred['betas']
            network.pred_cam = pred['cam']
            output = network.forward_smpl(**kwargs)
            pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

    # ========= Store results ========= #
    pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
    pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
    pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
    pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
    pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
    pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
    
    results['pose'] = pred_pose
    results['trans'] = pred_trans
    results['pose_world'] = pred_pose_world
    results['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
    results['betas'] = pred['betas'].cpu().squeeze(0).numpy()
    results['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
    
    if save_pkl:
        if args.run_smplify:
            if args.naive_intrinsics:
                pth = osp.join(output_pth, "smplify_naive_intrinsics.pkl")
                joblib.dump(results, pth)
                print("Save results to ", pth)
            else:
                pth = osp.join(output_pth, "smplify.pkl")
                joblib.dump(results, pth)
                print("Save results to ", pth)
        elif args.run_baseline:
            if args.use_gt_betas:
                pth = osp.join(output_pth, "baseline_gt_betas_at_once.pkl")
                joblib.dump(results, pth)
                print("Save results to ", pth)
            else:
                pth = osp.join(output_pth, "baseline.pkl")
                joblib.dump(results, pth)
                print("Save results to ", pth)
        else:
            pth = osp.join(output_pth, "eval.pkl")
            joblib.dump(results, pth)
            print("Save results to ", pth)

    align_and_compute_metrics(gt_data_path, pth, args, cfg)

if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    sequence_root = get_sequence_root(args)
    video_path = glob(os.path.join(sequence_root, "*.mov"))[0]

    # Output folder
    sequence = args.subject + "_" + args.sequence
    print(sequence)
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg,
        args,
        video_path, 
        output_pth, 
        network,
        args.save_pkl)
        
    logger.info('Done !')