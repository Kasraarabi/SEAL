#!/bin/bash

# Update package list
sudo apt-get update

# Install build-essential for compiling software
sudo apt-get install -y build-essential

# Install necessary libraries
sudo apt install -y libgl1-mesa-glx

# Install GCC using conda
conda install -y -c conda-forge gcc

# Install Python dependencies
pip install --force-reinstall -r requirements.txt