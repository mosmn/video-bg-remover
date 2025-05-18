#!/usr/bin/env python3
"""
Download the Robust Video Matting model file.
"""

import os
import sys
import requests
from tqdm import tqdm

def download_file(url, save_path):
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    if total_size == 0:
        print("Warning: Content length header not found. Progress bar may be inaccurate.")
        total_size = 1000000  # Assume ~1MB if unknown
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Downloaded size doesn't match expected size!")
        return False
    
    return True

def main():
    """Main function to download the RVM model."""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model options
    models = {
        "1": {
            "name": "MobileNetV3 (small, faster)",
            "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_small.pth",
            "filename": "rvm_mobilenetv3_small.pth"
        },
        "2": {
            "name": "MobileNetV3 (regular, recommended)",
            "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
            "filename": "rvm_mobilenetv3.pth"
        },
        "3": {
            "name": "ResNet50 (large, best quality)",
            "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
            "filename": "rvm_resnet50.pth"
        }
    }
    
    # Display model options
    print("Available RVM models:")
    for key, model in models.items():
        print(f"{key}. {model['name']}")
    
    # Get user choice
    choice = input("\nSelect model to download [2]: ").strip() or "2"
    
    if choice not in models:
        print(f"Invalid choice: {choice}. Defaulting to option 2.")
        choice = "2"
    
    selected_model = models[choice]
    model_path = os.path.join(models_dir, selected_model["filename"])
    
    # Check if model already exists
    if os.path.exists(model_path):
        overwrite = input(f"Model {selected_model['filename']} already exists. Redownload? (y/n) [n]: ").lower().strip() or "n"
        if overwrite != "y":
            print("Download canceled. Using existing model.")
            return
    
    # Download the model
    print(f"\nDownloading {selected_model['name']} model...")
    success = download_file(selected_model["url"], model_path)
    
    if success:
        print(f"\nModel downloaded successfully to: {model_path}")
    else:
        print("\nFailed to download model. Please try again or download manually.")
        print(f"Manual download URL: {selected_model['url']}")

if __name__ == "__main__":
    main()