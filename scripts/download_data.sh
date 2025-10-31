#!/bin/bash

# Create the data directory
mkdir -p data

# Download the sprite images
wget -O data/sprites_1788_16x16.npy 'https://huggingface.co/datasets/ashis-palai/sprites_image_dataset/resolve/a24918819843abc0d1bee75a239024415081a87d/sprites_1788_16x16.npy'

# Download the sprite labels
wget -O data/sprite_labels_nc_1788_16x16.npy 'https://huggingface.co/datasets/ashis-palai/sprites_image_dataset/resolve/a24918819843abc0d1bee75a239024415081a87d/sprite_labels_nc_1788_16x16.npy'

echo "Data downloaded and placed in data/"
