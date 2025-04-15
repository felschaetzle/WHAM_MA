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
from lib.models.utils import make_matching_plot_fast
from configs.config import parse_args

from scripts.custom_utils import get_sequence_root

def run(args):
    # Device config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load models
    superpoint = SuperPoint({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }).to(device).eval()

    superglue = SuperGlue({
        'weights': 'indoor'
    }).to(device).eval()

    root = get_sequence_root(args, gt=True)
    image_dir = Path(root) / "images"  # Adjust this path as needed
     # replace with your actual path
    image_paths = sorted(image_dir.glob("*.jpg"))  # or *.jpg if needed
    assert len(image_paths) >= 2, "Need at least 2 images to match"

    gt_data = joblib.load("/mnt/hdd/emdb_dataset/P4/36_outdoor_long_walk/P4_36_outdoor_long_walk_data.pkl")
    bbox = gt_data['bboxes']['bboxes']


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
    window = 50

    k_thresh = superpoint.config['keypoint_threshold']
    m_thresh = superglue.config['match_threshold']
    # Match every other frame against frame 0
    for i in range(len(image_paths)):
        if i % window == 0:
            n = i
            print("Processing reference frame", i)
            image_ref_tensor = load_and_preprocess_image(image_paths[i])
            frame0_data = process_frame(image_ref_tensor)

            # Filter out keypoints inside bounding boxes
            ref_frame_index = n  # This is your reference frame index
            bboxes_frame0 = bbox[ref_frame_index]  # shape: (num_people, 4)
            keypoints = frame0_data['keypoints'].cpu()

            # Remove keypoints inside bboxes
            filtered_kpts, valid_mask = filter_keypoints_outside_bboxes(keypoints, bboxes_frame0)
            frame0_data['keypoints'] = filtered_kpts.to(device)
            frame0_data['descriptors'] = frame0_data['descriptors'][:,valid_mask].to(device)
            frame0_data['scores'] = frame0_data['scores'][valid_mask].to(device)
            continue

        image_i_tensor = load_and_preprocess_image(image_paths[i])
        frame_i_data = process_frame(image_i_tensor)

        matches, confidences = match_frames(frame0_data, frame_i_data, image_ref_tensor, image_i_tensor)
        valid = matches > -1
        num_matches = valid.sum().item()

        confidences = confidences[valid]
        conf_mask = confidences > 0.7 
        confidences = confidences[conf_mask]
        num_matches = conf_mask.sum().item()
        # matches = matches[valid][conf_mask]
        print(f"[Frame {i:03d}] Matches with frame {n}: {conf_mask.sum()}. Confidence: {confidences.mean().item():.2f}, std: {confidences.std().item():.2f}")


        if num_matches > 0 and i % window == window - 1:
            kpts0 = frame0_data['keypoints'][valid][conf_mask].cpu().numpy()
            kpts1 = frame_i_data['keypoints'][matches][valid][conf_mask].cpu().numpy()
            colors = cm.jet(confidences.cpu().numpy())[:, :3] * 255  # RGB colors

            image0 = cv2.imread(str(image_paths[n]))
            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image1 = cv2.imread(str(image_paths[i]))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

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

            save_path = 'scripts/test/' + f"matches_{n:04}_{i:04}.png"
            cv2.imwrite(str(save_path), out)
        

if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True)
    run(args)