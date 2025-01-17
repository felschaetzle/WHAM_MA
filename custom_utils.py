import pickle
import joblib


import argparse
from glob import glob
import os
import os.path as osp

from configs import constants as _C

WHAM_OUTPUT = "output/emdb"


def open_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = joblib.load(f)
    return data

def get_sequence_root(args, gt=True):
    if gt:
        """Parse the path of the sequence to be visualized."""
        sequence_id = "{:0>2d}".format(int(args.sequence))
        print(os.path.join(_C.PATHS.EMDB_PTH,args.subject, sequence_id + "*"))
        candidates = glob(os.path.join(_C.PATHS.EMDB_PTH,args.subject, sequence_id + "*"))
        print(candidates)
        if len(candidates) == 0:
            raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
        elif len(candidates) > 1:
            raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
        return candidates[0]
    else:
        """Parse the path of the sequence to be visualized."""
        print
        sequence_id = "{:0>2d}".format(int(args.sequence))
        print(os.path.join(_C.PATHS.WHAM_OUTPUT,args.subject+"_"+sequence_id))
        candidates = glob(os.path.join(_C.PATHS.WHAM_OUTPUT,args.subject+"_"+sequence_id))
        print(candidates)
        if len(candidates) == 0:
            raise ValueError(f"Could not find sequence {args.sequence} for subject {args.subject}.")
        elif len(candidates) > 1:
            raise ValueError(f"Sequence ID {args.sequence}* for subject {args.subject} is ambiguous.")
        return candidates[0]

# wham = analyze_pkl("output/emdb/raw_short/wham_output.pkl")
# tracking = analyze_pkl("output/emdb/raw_short/tracking_results.pth")
# slam = open_pkl("output/emdb/P4_35/slam_results.pth")
# print("a")