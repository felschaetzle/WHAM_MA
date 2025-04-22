import os
import cv2
import torch

import sys
from pathlib import Path

# Add the parent directory of 'lib' to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from lib.models.superpoint import SuperPoint
from lib.models.superglue import SuperGlue

import joblib

import matplotlib.cm as cm
from pathlib import Path
import matplotlib.pyplot as plt
from lib.models.superglue_utils import make_matching_plot_fast
from configs.config import parse_args

from scripts.custom_utils import get_sequence_root
import numpy as np
from glob import glob
import pandas as pd

def clean_mask(mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return cleaned

def dilate_mask(mask, dilation_px=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return dilated

def filter_keypoints_outside_bboxes(keypoints, bbox):
    """Removes keypoints that fall inside any of the bounding boxes."""
    mask = torch.ones(len(keypoints), dtype=torch.bool)
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    inside = (
        (keypoints[:, 0] >= x1) & (keypoints[:, 0] <= x2) &
        (keypoints[:, 1] >= y1) & (keypoints[:, 1] <= y2)
    )
    mask &= ~inside
    return keypoints[mask], mask


def filter_keypoints_outside_mask(keypoints, mask):
    """
    Removes keypoints that fall inside the segmentation mask.

    Args:
        keypoints (torch.Tensor): shape (N, 2), keypoints in (x, y) format
        mask (np.ndarray or torch.Tensor): shape (H, W), binary mask (0/1 or bool)

    Returns:
        filtered_keypoints (torch.Tensor): keypoints not inside the mask
        valid_mask (torch.BoolTensor): mask indicating which keypoints were kept
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.bool()

    H, W = mask.shape
    x = keypoints[:, 0].long()
    y = keypoints[:, 1].long()

    # Clamp coordinates to ensure they're within the image bounds
    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)

    # Query mask at keypoint locations
    inside = mask[y, x]  # shape: (N,)
    valid_mask = ~inside  # keep those outside the mask

    return keypoints[valid_mask], valid_mask


def skew(t):
    """Return the skew-symmetric matrix of a vector t."""
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def get_fundamental_matrix(K, extrinsics_ref, extrinsics_frame):
    """
    Compute fundamental matrix from two extrinsics and intrinsics.

    extrinsics: [4x4] matrices, world-to-camera (i.e., [R | t])
    """
    # Get camera-to-world to compute relative motion from 0 to 49
    cam_to_world_ref = np.linalg.inv(extrinsics_ref)

    # Relative transform from cam0 to cam49
    rel_pose = extrinsics_frame @ cam_to_world_ref  # cam0 -> world -> cam49
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]

    # Compute fundamental matrix
    t_skew = skew(t)
    E = t_skew @ R  # Essential matrix
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F/np.linalg.norm(F)

def draw_epipolar_line(image, l, color=(0, 255, 0), thickness=2):
    a, b, c = l
    h, w = image.shape[:2]

    # Compute two endpoints of the line (at x=0 and x=w-1)
    if np.abs(b) > 1e-5:
        y0 = int((-c) / b)
        y1 = int((-a * (w - 1) - c) / b)
        pt1 = (0, y0)
        pt2 = (w - 1, y1)
    else:
        # Vertical line
        x = int(-c / a)
        pt1 = (x, 0)
        pt2 = (x, h - 1)

    img_with_line = image.copy()
    cv2.line(img_with_line, pt1, pt2, color=color, thickness=thickness)
    return img_with_line

def compute_epipolar_lines_batch(F, kpts0):
    kpts0_h = np.hstack([kpts0, np.ones((kpts0.shape[0], 1))])  # (n, 3)
    lines = (F @ kpts0_h.T).T  # (n, 3)
    # Normalize each line
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    return lines / norms

def epipolar_distances_batch(lines, kpts1):
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    u, v = kpts1[:, 0], kpts1[:, 1]
    return np.abs(a * u + b * v + c)

def run(args):

    root = get_sequence_root(args, gt=True)
    image_dir = Path(root) / "images"  # Adjust this path as needed
     # replace with your actual path
    image_paths = sorted(image_dir.glob("*.jpg"))  # or *.jpg if needed
    assert len(image_paths) >= 2, "Need at least 2 images to match"

    gt_data_pth = glob(os.path.join(root,"*.pkl"))[0]


    # Device config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models
    superpoint = SuperPoint({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }).to(device).eval()


    if "outdoor" in gt_data_pth:
        superglue = SuperGlue({
            'weights': 'outdoor'
        }).to(device).eval()
    elif "indoor" in gt_data_pth:
        superglue = SuperGlue({
            'weights': 'indoor'
        }).to(device).eval() 
    else:
        print("Not indoor and not outdoor!")
        return
    
    gt_data = joblib.load(gt_data_pth)
    bbox = gt_data['bboxes']['bboxes']
    gt_ext = gt_data['camera']['extrinsics']
    K = gt_data['camera']['intrinsics'] 
    frame_mask = gt_data['good_frames_mask']


    wham_root = get_sequence_root(args, False)
 
    baseline = joblib.load(wham_root+'/baseline_gt_betas.pkl')
    dpvo_ext = baseline['dpvo_extrinsics']

    if args.smooth_wham_cam:
        wham_ext = baseline['wham_cam']
    else:
        wham_ext = baseline['wham_cam_init']

    mask_dir = Path(f"/home/felix/sam2/emdb_segmentation/{args.sequence}") # Numpy file with shape (n, h, w)
    mask_paths = sorted(mask_dir.glob("*.png"))           # shape (n, h, w)
  
    # Load and preprocess
    def load_and_preprocess_image(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"Could not read image: {path}"
        image = image.astype('float32') / 255.0
        tensor = torch.from_numpy(image)[None, None].to(device)  # (1, 1, H, W)
        return tensor

    # Extract SuperPoint features
    @torch.no_grad()
    def process_frame(image_tensor):
        data = {'image': image_tensor}
        pred = superpoint(data)
        return {
            'keypoints': pred['keypoints'][0],
            'descriptors': pred['descriptors'][0],
            'scores': pred['scores'][0]
        }

    # Match features with SuperGlue
    @torch.no_grad()
    def match_frames(f0, f1, image0, image1):
        data = {
            'keypoints0': f0['keypoints'][None],
            'keypoints1': f1['keypoints'][None],
            'descriptors0': f0['descriptors'][None],
            'descriptors1': f1['descriptors'][None],
            'scores0': f0['scores'][None],
            'scores1': f1['scores'][None],
            'image0': image0,
            'image1': image1
        }
        for k in data:
            data[k] = data[k].to(device)
        pred = superglue(data)
        return pred['matches0'][0], pred['matching_scores0'][0]

    # ---- Main Matching Loop ----
    n = 0
    n_wham = 0
    window = 40

    k_thresh = superpoint.config['keypoint_threshold']
    m_thresh = superglue.config['match_threshold']
    # Match every other frame against frame 0

    gt_epi_error = []
    dpvo_epi_error = []
    wham_epi_error = []
    index = []

    num_keypoints = []
    conf = []

    frames = np.array(range(len(image_paths)))[frame_mask]
    
    switch_keyframe = False

    for i, elm in enumerate(frames):
        # if i > 900:
        #     print('h')
            # break


        # keyframe selector:

        if i % window == 0:
            n = elm
            n_wham = i

            image_ref_tensor = load_and_preprocess_image(image_paths[elm])
            frame0_data = process_frame(image_ref_tensor)

            # Filter out keypoints inside bounding boxes
            ref_frame_index = n  # This is your reference frame index
            bboxes_frame0 = bbox[ref_frame_index]  # shape: (num_people, 4)
            keypoints = frame0_data['keypoints'].cpu()


            mask_path = mask_paths[elm]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            cleaned_mask = clean_mask(mask)
            offset_mask = dilate_mask(cleaned_mask)


            # Remove keypoints inside bboxes
            # filtered_kpts, valid_mask = filter_keypoints_outside_bboxes(keypoints, bboxes_frame0)
            filtered_kpts, valid_mask = filter_keypoints_outside_mask(keypoints, offset_mask)
            frame0_data['keypoints'] = filtered_kpts.to(device)
            frame0_data['descriptors'] = frame0_data['descriptors'][:,valid_mask].to(device)
            frame0_data['scores'] = frame0_data['scores'][valid_mask].to(device)
            continue

        if i % window == window - 1:
            image_i_tensor = load_and_preprocess_image(image_paths[elm])
            frame_i_data = process_frame(image_i_tensor)

            matches, confidences = match_frames(frame0_data, frame_i_data, image_ref_tensor, image_i_tensor)
            valid = matches > -1
            num_matches = valid.sum().item()

            confidences = confidences[valid]
            conf_mask = confidences > 0.7
            confidences = confidences[conf_mask]
            num_matches = conf_mask.sum().item()
            num_keypoints.append(num_matches)
            if num_matches == 0:
                conf.append(0)
            else:
                conf.append(confidences.mean().item()*100)

            if  num_matches > 0:
                kpts0 = frame0_data['keypoints'][valid][conf_mask].cpu().numpy()
                kpts1 = frame_i_data['keypoints'][matches][valid][conf_mask].cpu().numpy()
  
                colors = cm.jet(confidences.cpu().numpy())[:, :3] * 255  # RGB colors

                image0 = cv2.imread(str(image_paths[n]))
                image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                image1 = cv2.imread(str(image_paths[i]))
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

                gt_F = get_fundamental_matrix(K, gt_ext[n], gt_ext[elm])
                gt_lines = compute_epipolar_lines_batch(gt_F, kpts0)
                gt_error = epipolar_distances_batch(gt_lines, kpts1)
                
                dpvo_F = get_fundamental_matrix(K, dpvo_ext[n], dpvo_ext[elm])
                dpvo_lines = compute_epipolar_lines_batch(dpvo_F, kpts0)
                dpvo_error = epipolar_distances_batch(dpvo_lines, kpts1)

                wham_F = get_fundamental_matrix(K, wham_ext[n_wham], wham_ext[i])
                wham_lines = compute_epipolar_lines_batch(wham_F, kpts0)
                wham_error = epipolar_distances_batch(wham_lines, kpts1)

                gt_epi_error.append(gt_error.mean())
                dpvo_epi_error.append(dpvo_error.mean())
                wham_epi_error.append(wham_error.mean())
                index.append(elm)

                # print("Epipolar distance GT, DPVO, WHAM:", gt_error.mean().item(), dpvo_error.mean().item(), wham_error.mean())
                
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh)
                ]

                out = make_matching_plot_fast(
                    image0, image1, frame0_data['keypoints'].cpu().numpy(),
                    frame_i_data['keypoints'].cpu().numpy(), kpts0, kpts1,
                    colors, small_text, 
                    path=None, show_keypoints=False,
                    small_text=[f'Matches: {num_matches}', f'Frame pair: {n}-{i}']
                )

                save_path = 'output/tracker/images/' + f"matches_{n:04}_{i:04}.png"
                cv2.imwrite(str(save_path), out)
                # print(f"[Frame {i:03d}] Matches with frame {n}: {conf_mask.sum()}. Confidence: {confidences.mean().item():.2f}, std: {confidences.std().item():.2f}")
            else:
                gt_epi_error.append(0)
                dpvo_epi_error.append(0)
                wham_epi_error.append(0)
                index.append(elm)
                print('no matches for frame', i)

    pth_root = f"output/tracker/{args.subject}_{args.sequence}"
    if args.smooth_wham_cam:
        path = f"{pth_root}_smooth_wham_cam"
    else:
        path = pth_root


    # === Plot 1: Epipolar Errors === #
    p = path+"_epipolor_distance.png"
    plt.figure(figsize=(10, 5))
    plt.plot(index, gt_epi_error, label='GT', marker='o')
    plt.plot(index, dpvo_epi_error, label='DPVO', marker='o')
    plt.plot(index, wham_epi_error, label='WHAM', marker='o')

    plt.xlabel('Frame Index')
    plt.ylabel('Epipolar Error (pixels)')
    plt.title('Epipolar Error Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p)
    print("Epipolar error plot saved to", p)

    # === Plot 2: Confidence and Keypoints === #
    p = path +"_keypoints.png"
    plt.figure(figsize=(10, 5))
    plt.plot(index, conf, label="Confidence", marker="o")
    plt.plot(index, num_keypoints, label="# Keypoints", marker="o")

    plt.xlabel('Frame Index')
    plt.ylabel('Confidence / Keypoints')
    plt.title('Detection Confidence and Keypoints Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(p)
    print("Confidence/keypoints plot saved to", p)


    df = pd.DataFrame({
        'frame_index': index,          # frame indices (same length as num_keypoints)
        'num_keypoints': num_keypoints,
        'confidence': conf,
        'gt_epipolar_loss': gt_epi_error,
        'dpvo_epipolar_loss': dpvo_epi_error,
        'wham_epipolar_loss': wham_epi_error
    })

    # Build file path (matching the naming from the plots)
    csv_path = f"{path}.csv"
    df.to_csv(csv_path, index=False)

    print("Saved num_keypoints to", csv_path)

    print('done')
if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True)
    run(args)