"""
Utility functions for logging, plotting, and saving.
"""

import logging
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image, make_grid
from src import config

def setup_logging():
    """Configures the logging for the project."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Basic config for console (INFO level)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    
    # File handler for detailed logs (DEBUG level)
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to the root logger
    logging.getLogger().addHandler(file_handler)
    
    logging.info("Logging setup complete. Logs will be saved to %s", config.LOG_FILE)

def plot_loss(train_losses, val_losses, save_path):
    """Plots and saves the training and validation loss curves."""
    epochs = list(range(1, len(val_losses) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of Training UNet Diffusion Model over Sprites Dataset")
    plt.legend()
    plt.grid()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info("Loss plot saved to %s", save_path)

def save_samples(images, save_path, n_rows=4, normalize=True):
    """Saves a grid of generated images."""
    if normalize:
        # Normalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
    
    grid = make_grid(images, nrow=n_rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(grid, save_path)
    logging.info("Sample grid saved to %s", save_path)
