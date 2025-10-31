"""
Custom Dataset and DataLoader for the Sprites dataset.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src import config

class CustomDataset(Dataset):
    """Sprites Dataset."""

    def __init__(self, sprites, labels, transform, null_context=False):
        self.transform = transform
        self.null_context = null_context
        self.sprites = sprites
        self.labels = labels

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        image = self.transform(self.sprites[idx])
        
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = torch.tensor(self.labels[idx]).to(torch.int64)
            
        return (image, label)

def get_dataloaders():
    """Loads data, creates datasets, and returns dataloaders."""
    
    try:
        sprites = np.load(config.SPRITES_PATH)
        labels = np.load(config.LABELS_PATH)
    except FileNotFoundError:
        logging.error("Data files not found. Please run 'bash scripts/download_data.sh'")
        raise
        
    logging.info(f"Loaded sprites shape: {sprites.shape}")
    logging.info(f"Loaded labels shape: {labels.shape}")

    # Split data into train and validation
    train_sprites, val_sprites, train_labels, val_labels = train_test_split(
        sprites, labels, test_size=config.VAL_SPLIT, random_state=42
    )
    
    logging.info(f"Training data: {len(train_sprites)} samples")
    logging.info(f"Validation data: {len(val_sprites)} samples")

    transform = transforms.Compose([
        transforms.ToTensor(),       # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # range [-1,1]
    ])

    train_dataset = CustomDataset(train_sprites, train_labels, transform, null_context=False)
    val_dataset = CustomDataset(val_sprites, val_labels, transform, null_context=False)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )

    return train_dataloader, val_dataloader, val_dataset
