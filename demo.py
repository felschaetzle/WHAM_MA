import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

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

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False


def run(cfg,
        video,
        output_pth,
        network,
        run_global=True,
        save_pkl=False,
        visualize=False,
        gt_bb_kp=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    print(args.gt_intrinsics, args.gt_extrinsics)

    run_preproc = True
    if not args.gt_intrinsics:
        if osp.exists(osp.join(output_pth, 'slam_results.pth')):
            run_preproc = False
        calib = None
    else:
        calib = "output/emdb/"+ args.subject + "_" + args.sequence + "/gt_intrinsics.txt"
        if osp.exists(osp.join(output_pth, 'slam_results_gt_intrinsics.pth')):
            run_preproc = False

    # Preprocess
    with torch.no_grad():
        if run_preproc:
            
            # detector = DetectionModel(cfg.DEVICE.lower())
            # extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                # detector.track(img, fps, length)
                
                # SLAM
                if slam is not None: 
                    slam.track()
                
                bar.next()

            # tracking_results = detector.process(fps)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            # tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            if not args.gt_intrinsics:
                # joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
                joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
                logger.info(f'Save processed data at {output_pth}')
            else:
                # joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results_gt_intrinsics.pth'))
                joblib.dump(slam_results, osp.join(output_pth, 'slam_results_gt_intrinsics.pth'))
                logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            if not args.gt_intrinsics:
                # tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
                slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
                logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
            else:
                # tracking_results = joblib.load(osp.join(output_pth, 'tracking_results_gt_intrinsics.pth'))
                slam_results = joblib.load(osp.join(output_pth, 'slam_results_gt_intrinsics.pth'))
                logger.info(f'Already processed data exists at {output_pth} ! Load the data .')

    eval_loader = setup_eval_dataloader(cfg, 'emdb', args.eval_split, cfg.MODEL.BACKBONE)
    emdb_sequence_index = find_substring(args.subject+"_"+args.sequence, eval_loader.dataset.labels['vid'])
    if emdb_sequence_index is None:
        logger.error(f"Sequence {args.subject}_{args.sequence} not found in the emdb2 dataset. Not usefull for global trajectory.")
        return
    # slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
    cam_angvel = convert_dpvo_to_cam_angvel(slam_results, fps).to(cfg.DEVICE).unsqueeze(0)
    print("Loading data from eval loader")

    if args.gt_intrinsics:
        calib_data = np.loadtxt(calib, delimiter=" ")
        fx, fy, cx, cy = calib_data[:4]
        gt_intrinsics = np.eye(3)
        gt_intrinsics[0,0] = fx
        gt_intrinsics[0,2] = cx
        gt_intrinsics[1,1] = fy
        gt_intrinsics[1,2] = cy
        gt_intrinsics = torch.tensor(gt_intrinsics).float().to(cfg.DEVICE).unsqueeze(0)
        print("GT intrinsics")
        print(gt_intrinsics)

    sequence_root = get_sequence_root(args)
    gt_data_path = glob(os.path.join(sequence_root, "*_data.pkl"))[0]
    gt_data = joblib.load(gt_data_path)
    gt_extrinsics = gt_data["camera"]["extrinsics"]
    gt_cam_pose = np.linalg.inv(gt_extrinsics)
    gt_cam_pose_rot = R.from_matrix(gt_cam_pose[:,:3,:3]).as_quat()
    gt_cam_pose = np.concatenate([gt_cam_pose[:,:3,3], gt_cam_pose_rot], axis=1)

    results = defaultdict(dict)
    
    with torch.no_grad():
        # Forward pass with flipped input
        flipped_batch = eval_loader.dataset.load_data(emdb_sequence_index, flip=True)
        x, inits, features, kwargs, gt = prepare_batch(flipped_batch, cfg.DEVICE)
        if args.gt_extrinsics and args.gt_intrinsics:
            print("Use GT intrinsics and GT extrinsics")
            kwargs['cam_intrinsics'] = gt_intrinsics.unsqueeze(0)

        elif args.gt_intrinsics:
            print("Use GT intrinsics and use DPVO with GT intrinsics")
            kwargs['cam_intrinsics'] = gt_intrinsics.unsqueeze(0)
            kwargs['cam_angvel'] = cam_angvel
            
        else:
            print("Don't replace intrinsics and use DPVO with GT intrinsics")
            kwargs['cam_angvel'] = cam_angvel

        flipped_pred = network(x, inits, features, return_y_up=True, **kwargs)
        
        # Forward pass with normal input
        flipped_batch = eval_loader.dataset.load_data(emdb_sequence_index, flip=False)
        x, inits, features, kwargs, gt = prepare_batch(flipped_batch, cfg.DEVICE)
        if args.gt_extrinsics and args.gt_intrinsics:
            kwargs['cam_intrinsics'] = gt_intrinsics.unsqueeze(0)

        elif args.gt_intrinsics:
            kwargs['cam_intrinsics'] = gt_intrinsics.unsqueeze(0)
            kwargs['cam_angvel'] = cam_angvel
            
        else:
            kwargs['cam_angvel'] = cam_angvel

        pred = network(x, inits, features, return_y_up=True, **kwargs)

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
        cam_angvel = kwargs['cam_angvel']
        pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

    if args.run_smplify:
        # convert gt extrinsics to torch tensor
        kwargs["gt_extrinsics"] = torch.tensor(gt_extrinsics).float().to(cfg.DEVICE).unsqueeze(0)
        input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index][1:,:,:].to(cfg.DEVICE)
        # smplify = CustomSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
        # pred = smplify.fit(pred, input_keypoints, **kwargs)
        pred = progressive_global_translation_optimization(
            pred, input_keypoints, kwargs['bbox'],
            kwargs['gt_extrinsics'], kwargs['cam_intrinsics'],
            smpl, cfg.DEVICE, length, kwargs['res'][0,:])

        # with torch.no_grad():
        #     network.pred_pose = pred['pose']
        #     network.pred_shape = pred['betas']
        #     network.pred_cam = pred['cam']
        #     output = network.forward_smpl(**kwargs)
        #     pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        # smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
        # input_keypoints = eval_loader.dataset.labels['kp2d'][emdb_sequence_index].cpu().numpy()
        # input_keypoints = input_keypoints[1:,:,:]
        # pred = smplify.fit(pred, input_keypoints, **kwargs)
        
        # with torch.no_grad():
        #     network.pred_pose = pred['pose']
        #     network.pred_shape = pred['betas']
        #     network.pred_cam = pred['cam']
        #     output = network.forward_smpl(**kwargs)
        #     pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)

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
        if args.gt_extrinsics and args.gt_intrinsics:
            if args.run_smplify:
                joblib.dump(results, osp.join(output_pth, "wham_output_gt_camera.pkl"))
            else:
                joblib.dump(results, osp.join(output_pth, "wham_output_gt_camera_wo_SMPLify.pkl"))
        elif args.gt_intrinsics:
            joblib.dump(results, osp.join(output_pth, "wham_output_gt_intrinsics.pkl"))
        else:
            joblib.dump(results, osp.join(output_pth, "wham_output_DPVO.pkl"))
     
    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            if args.gt_intrinsics:
                focal_len = gt_intrinsics[0, 0, 0]
            else:
                focal_len = None
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global, focal_length=focal_len)
        

if __name__ == '__main__':
    subject_id = _C.subject_id
    sequence_id = _C.sequence_id

    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_intrinsics", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, 
                        help="Use GT intrinsics (True/False)")

    parser.add_argument("--gt_extrinsics", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, 
                        help="Use ground truth camera pose (True/False)")

    parser.add_argument('--video', type=str, 
                        default='/mnt/hdd/emdb_dataset/P5/40_indoor_walk_big_circle/raw.mov', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default=_C.PATHS.WHAM_OUTPUT, 
                        help='output folder to write results')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true', default=True,
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true', default=True,
                        help='Run Temporal SMPLify for post processing')
    
    parser.add_argument("--subject", type=str, default=subject_id, help="The subject ID, P0 - P9.")

    parser.add_argument(
        "--sequence",
        type=str,
        default=sequence_id,
        help="The sequence ID. This can be any unambiguous prefix of the sequence's name, i.e. for the "
        "sequence '66_outdoor_rom' it could be '66' or any longer prefix including the full name.",
    )

    parser.add_argument(
        "--eval-split", type=str, default='2', help="Evaluation data split")

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    sequence_root = get_sequence_root(args)
    video_path = glob(os.path.join(sequence_root, "*.mov"))[0]

    # Output folder
    sequence = args.subject + "_" + args.sequence
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    run(cfg, 
        video_path, 
        output_pth, 
        network,
        run_global=not args.estimate_local_only, 
        save_pkl=args.save_pkl,
        visualize=args.visualize)
        
    print()
    logger.info('Done !')