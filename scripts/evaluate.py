"""
Script to generate images and compute the FID score.
"""

import argparse
import logging
import os
import shutil
import torch
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src import utils
from src.data_loader import get_dataloaders
from src.model import Unet
from src.diffusion import DiffusionScheduler
from scripts.sample import generate_samples # Re-use the generation logic
from torchvision.utils import save_image

def save_real_images(dataset, n_samples, save_dir):
    """Saves real images from the dataset for FID comparison."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    logging.info(f"Saving {n_samples} real images to {save_dir}...")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (img, _) in enumerate(data_loader):
        if i >= n_samples:
            break
        # Normalize from [-1, 1] to [0, 1] for saving
        img = (img + 1) / 2
        save_image(img, os.path.join(save_dir, f"real_{i}.png"))
    logging.info("Real images saved.")

def save_generated_images(model, scheduler, n_samples, batch_size, save_dir, device, method, eta, n_ddim_steps):
    """Generates and saves fake images for FID comparison."""
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    logging.info(f"Generating {n_samples} fake images to {save_dir}...")
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    img_id = 0
    
    for _ in tqdm(range(n_batches), desc="Generating FID Images"):
        n_batch_samples = min(batch_size, n_samples - img_id)
        if n_batch_samples == 0:
            break
            
        images = generate_samples(model, scheduler, n_batch_samples, device, method, eta, n_ddim_steps)
        
        # Normalize from [-1, 1] to [0, 1] for saving
        images = (images + 1) / 2
        
        for img in images:
            save_image(img, os.path.join(save_dir, f"generated_{img_id}.png"))
            img_id += 1
            
    logging.info("Generated images saved.")

def main(args):
    """Main evaluation routine."""
    utils.setup_logging()
    logging.info("Starting evaluation script...")
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Get validation dataset (to save real images from)
    _, _, val_dataset = get_dataloaders()
    
    # Save real images
    save_real_images(val_dataset, args.n_samples, config.EVAL_REAL_DIR)
    
    # Load model
    model = Unet(
        in_channels=config.IN_CHANNELS,
        n_feat=config.N_FEAT,
        n_cfeat=config.N_CFEAT,
        height=config.HEIGHT
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Model loaded from {args.model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found at {args.model_path}. Please train the model first.")
        return

    # Initialize scheduler
    scheduler = DiffusionScheduler(
        timesteps=config.TIMESTEPS,
        beta1=config.BETA_1,
        beta2=config.BETA_2,
        device=device
    )
    
    # Generate and save fake images
    save_generated_images(
        model, 
        scheduler, 
        args.n_samples, 
        args.batch_size, 
        config.EVAL_GEN_DIR, 
        device, 
        args.method,
        args.eta,
        args.n_ddim_steps
    )
    
    # Compute FID score
    logging.info("Calculating FID score...")
    fid = fid_score.calculate_fid_given_paths(
        [config.EVAL_REAL_DIR, config.EVAL_GEN_DIR],
        batch_size=config.FID_BATCH_SIZE,
        device=device,
        dims=2048
    )
    
    logging.info(f"*** FID Score: {fid} ***")
    
    # Clean up
    if not args.keep_images:
        logging.info("Cleaning up image directories...")
        shutil.rmtree(config.EVAL_REAL_DIR)
        shutil.rmtree(config.EVAL_GEN_DIR)
        
    logging.info("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPM model using FID.")
    parser.add_argument("--n-samples", type=int, default=config.FID_N_SAMPLES, help="Number of real/generated samples for FID.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for image generation.")
    parser.add_argument("--model-path", type=str, default=config.MODEL_SAVE_PATH, help="Path to the trained model.")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use (cuda or cpu).")
    parser.add_argument("--keep-images", action="store_true", help="Keep the generated images after FID calculation.")
    parser.add_action('store', type=str, dest="method", default="ddpm", choices=["ddpm", "ddim"], help="Sampling method: ddpm or ddim.")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta value for DDIM sampling (0.0=deterministic).")
    parser.add_argument("--n-ddim-steps", type=int, default=50, help="Number of steps for DDIM sampling.")

    args = parser.parse_args()
    main(args)
