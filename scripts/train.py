"""
Main training script for the DDPM.
"""

import argparse
import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src import utils
from src.data_loader import get_dataloaders
from src.model import Unet
from src.diffusion import DiffusionScheduler

def train_epoch(model, scheduler, dataloader, optimizer, device):
    """Performs one training epoch."""
    model.train()
    train_loss = 0.0
    
    for x0, _ in tqdm(dataloader, desc="Training"):
        x0 = x0.to(device)
        
        # Sample random timesteps
        t = torch.randint(0, config.TIMESTEPS, (x0.size(0),), device=device).long()

        # Perturb the input (forward process)
        xt, noise = scheduler.add_noise(x0, t)

        # Predict the noise
        noise_pred = model(xt, t.unsqueeze(-1).float())

        # Compute the loss
        loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(dataloader)

def val_epoch(model, scheduler, dataloader, device):
    """Performs one validation epoch."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x0, _ in tqdm(dataloader, desc="Validation"):
            x0 = x0.to(device)
            t = torch.randint(0, config.TIMESTEPS, (x0.size(0),), device=device).long()
            xt, noise = scheduler.add_noise(x0, t)
            noise_pred = model(xt, t.unsqueeze(-1).float())
            loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()
            
    return val_loss / len(dataloader)

def main(args):
    """Main training routine."""
    utils.setup_logging()
    logging.info("Starting training script...")
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    train_loader, val_loader, _ = get_dataloaders()
    
    # Initialize model, scheduler, and optimizer
    model = Unet(
        in_channels=config.IN_CHANNELS,
        n_feat=config.N_FEAT,
        n_cfeat=config.N_CFEAT,
        height=config.HEIGHT
    ).to(device)
    
    scheduler = DiffusionScheduler(
        timesteps=config.TIMESTEPS,
        beta1=config.BETA_1,
        beta2=config.BETA_2,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_epoch(model, scheduler, train_loader, optimizer, device)
        val_loss = val_epoch(model, scheduler, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logging.info(f"New best model saved to {config.MODEL_SAVE_PATH}")
            
    # Save loss plot
    plot_path = f"{config.OUTPUT_DIR}/loss_plot.png"
    utils.plot_loss(train_losses, val_losses, plot_path)
    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM model on Sprites dataset.")
    parser.add_argument("--epochs", type=int, default=config.N_EPOCH, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=config.LRATE, help="Learning rate.")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Device to use (cuda or cpu).")
    
    args = parser.parse_args()
    
    # Update config (in case args are used, though config.py is primary)
    config.N_EPOCH = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LRATE = args.lr
    
    main(args)
