"""
Script to generate and save sample images using the trained DDPM.
"""

import argparse
import logging
import os
import torch
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src import utils
from src.model import Unet
from src.diffusion import DiffusionScheduler

@torch.no_grad()
def generate_samples(model, scheduler, n_samples, device, method, eta, n_ddim_steps):
    """Generates images using the reverse diffusion process."""
    model.eval()
    
    xt = torch.randn((n_samples, config.IN_CHANNELS, config.HEIGHT, config.HEIGHT)).to(device)
    
    if method == "ddpm":
        logging.info(f"Starting DDPM sampling for {config.TIMESTEPS} steps...")
        timesteps = list(range(config.TIMESTEPS - 1, -1, -1))
    
    elif method == "ddim":
        logging.info(f"Starting DDIM sampling for {n_ddim_steps} steps (eta={eta})...")
        timesteps = torch.linspace(config.TIMESTEPS - 1, 0, n_ddim_steps).long().tolist()

    
    for i in tqdm(range(len(timesteps)), desc="Sampling"):
        t = timesteps[i]
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(xt, t_tensor.unsqueeze(-1).float())
        
        if method == "ddpm":
            xt = scheduler.p_xt_ddpm(xt, t_tensor, predicted_noise)
        
        elif method == "ddim":
            t_minus_1 = timesteps[i+1] if i < len(timesteps) - 1 else -1
            t_minus_1_tensor = torch.full((n_samples,), t_minus_1, device=device, dtype=torch.long)
            xt = scheduler.p_xt_ddim(xt, t_tensor, t_minus_1_tensor, predicted_noise, eta)
            
    return xt

def main(args):
    """Main sampling routine."""
    utils.setup_logging()
    logging.info("Starting sampling script...")
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
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
    
    # Generate samples
    images = generate_samples(
        model, 
        scheduler, 
        args.n_samples, 
        device, 
        args.method, 
        args.eta,
        args.n_ddim_steps
    )
    
    # Save samples
    save_path = f"{config.SAMPLE_DIR}/samples_{args.method}_eta{args.eta}.png"
    utils.save_samples(images, save_path, n_rows=int(args.n_samples**0.5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained DDPM model.")
    parser.add_argument("--n-samples", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--model-path", type=str, default=config.MODEL_SAVE_PATH, help="Path to the trained model.")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use (cuda or cpu).")
    parser.add_action('store', type=str, dest="method", default="ddpm", choices=["ddpm", "ddim"], help="Sampling method: ddpm or ddim.")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta value for DDIM sampling (0.0=deterministic).")
    parser.add_argument("--n-ddim-steps", type=int, default=50, help="Number of steps for DDIM sampling.")
    
    args = parser.parse_args()
    main(args)
