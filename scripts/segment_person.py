import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from PIL import Image
import joblib
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/home/felix/WHAM_MA')
from scripts.custom_utils import get_sequence_root
from configs.config import parse_args

from pathlib import Path
from glob import glob
import cv2
from tqdm import tqdm

import imageio

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

keep_idxs = [
    12, 15,       # Head
    # 16, 17,   # L/R shoulder
    18, 19,   # L/R elbow
    22, 23,   # L/R hand
     1,  2,   # L/R hip
     4,  5,   # L/R knee
     7,  8    # L/R ankle
]

def run(args):
    root = get_sequence_root(args, gt=True)
    image_dir = Path(root) / "images" 
    image_paths = sorted(image_dir.glob("*.jpg"))  # or *.jpg if needed
    gt_data_pth = glob(os.path.join(root,"*.pkl"))[0]

    gt_data = joblib.load(gt_data_pth)
    kps = gt_data['kp2d']

    H, W = np.asarray(Image.open(image_paths[0]).convert("RGB")).shape[:2]

    print(f"Run sequence {args.sequence}")
    for n, elm in tqdm(enumerate(image_paths), total=kps.shape[0]):
        # if n != 979:
        #     continue
        # if n < 900 or n > 1050:
        #     continue


        img = np.asarray(Image.open(elm).convert("RGB")).copy()
        predictor.set_image(img)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            kp = kps[n][keep_idxs]
            xs, ys = kp[:,0], kp[:,1]
            in_bounds = (xs >= 10) & (xs < W-10) & (ys >= 10) & (ys < H-10)
            kp = kp[in_bounds]
            masks, _, logits = predictor.predict(point_coords=kp, point_labels=np.ones(kp.shape[0]), multimask_output=False)

            for i in range(3):
                # iterate 3 times to refine the masks
                masks, _, logits = predictor.predict(mask_input=logits, multimask_output=False)

            masks = masks.reshape((H,W))


        path = f"emdb_segmentation/{args.sequence}"
        os.makedirs(path, exist_ok=True)

        # mask_rgb = np.stack([(masks * 255).astype(np.uint8), (masks * 255).astype(np.uint8), (masks * 255).astype(np.uint8)], axis=-1)  # H×W×3
        # for (x, y) in kp:
        #     cv2.circle(
        #         mask_rgb, 
        #         (int(x), int(y)),   # pixel coords
        #         radius=5,           # adjust size as you like
        #         color=(0, 255, 0),  # green
        #         thickness=-1        # filled circle
        #     )

        # imageio.imwrite(f"test_mask_{n:05d}.png", mask_rgb)   


        imageio.imwrite(f"{path}/mask_{n:05d}.png", (masks * 255).astype(np.uint8))   

if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True)
    run(args)
