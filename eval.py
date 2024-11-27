from cleanfid import fid
import cv2
import os
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import argparse
import math
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.io
import torch


real_folder = ""
fake_folder = ""

score = fid.compute_fid(real_folder, fake_folder)
