"""
Configuration file for all hyperparameters and settings.
"""

import torch

# Diffusion hyperparameters
TIMESTEPS = 1000
BETA_1 = 1e-4
BETA_2 = 0.02

# Network hyperparameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
N_FEAT = 128 
N_CFEAT = 5  # Context vector size
HEIGHT = 16  # 16x16 image
IN_CHANNELS = 3

# Training hyperparameters
BATCH_SIZE = 128
N_EPOCH = 100
LRATE = 1e-3
VAL_SPLIT = 0.1

# Data and Output paths
DATA_DIR = "data"
SPRITES_PATH = f"{DATA_DIR}/sprites_1788_16x16.npy"
LABELS_PATH = f"{DATA_DIR}/sprite_labels_nc_1788_16x16.npy"

OUTPUT_DIR = "outputs"
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/ddpm_sprites.log"
MODEL_DIR = f"{OUTPUT_DIR}/models"
MODEL_NAME = "ddpm_sprite_best.pth"
MODEL_SAVE_PATH = f"{MODEL_DIR}/{MODEL_NAME}"

# Sampling and Evaluation
SAMPLE_DIR = f"{OUTPUT_DIR}/samples"
EVAL_DIR = f"{OUTPUT_DIR}/eval"
EVAL_REAL_DIR = f"{EVAL_DIR}/real"
EVAL_GEN_DIR = f"{EVAL_DIR}/generated"
FID_N_SAMPLES = 3000
FID_BATCH_SIZE = 50
