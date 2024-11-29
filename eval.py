from ast import arg
from cleanfid import fid
import cv2
import os
import numpy as np
# from skimage.metrics import mean_squared_error
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity as ssim
import argparse
import math
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.io
import torch
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--real_folder", type=str, required=True, help="Path to the real image folder"
    )
    parser.add_argument(
        "--fake_folder", type=str, required=True, help="Path to the fake image folder"
    )
    return parser


parser = get_parser()
args = parser.parse_args()


real_folder = args.real_folder
fake_folder = args.fake_folder

score = fid.compute_fid(real_folder, fake_folder)
print(f"FID score: {score:.5f}")
