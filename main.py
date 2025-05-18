#!/usr/bin/env python3
"""
Video Background Removal Tool using Robust Video Matting (RVM).
Optimized for MacBook Air M1 using Apple's Metal Performance Shaders (MPS).
"""

import os
import sys
import time
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Union, List
from utils import load_model, process_frame, get_background

def validate_file(file_path: str) -> bool:
    """Check if file exists and is accessible."""
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

def process_video(input_path: str, 
                 output_path: str, 
                 model_path: str,
                 output_type: str = 'video',
                 bg_color: Optional[Tuple[int, int, int]] = None,
                 downsample_ratio: float = 1.0,
                 device: str = 'auto') -> None:
    """
    Process a video file to remove background.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        model_path: Path to RVM model file
        output_type: Output format ('video', 'rgba', 'frames')
        bg_color: Background color as RGB tuple
        downsample_ratio: Downsample ratio for faster processing
        device: Computation device ('cpu', 'cuda', 'mps', or 'auto')
    """
    if not validate_file(input_path):
        print(f"Error: Input file '{input_path}' not found or not readable.")
        return
    
    if not validate_file(model_path):
        print(f"Error: Model file '{model_path}' not found or not readable.")
        return
    
    # Load video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    if output_type == 'frames':
        # Create output directory for frames
        os.makedirs(output_path, exist_ok=True)
        writer = None
    else:
        # Determine output format
        if output_type == 'rgba':
            # RGBA output for transparency
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Note: limited support for alpha
            channels = 4
        else:
            # Regular RGB output with background
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            channels = 3
        
        # Create video writer
        writer = cv2.VideoWriter(
            output_path, 
            fourcc,
            fps,
            (width, height),
            True
        )
        if not writer.isOpened():
            print(f"Error: Could not create output video file {output_path}")
            cap.release()
            return
    
    # Load model
    print("Loading RVM model...")
    model, model_info = load_model(model_path, device)
    device = model_info['device']
    print(f"Model loaded - Type: {model_info['model_type']}, Device: {device}")
    
    # Determine background type
    bg_type = 'none' if output_type == 'rgba' else 'color'
    
    # Create background if needed
    if bg_type == 'color' and bg_color is None:
        # Default to green background if none specified
        bg_color = (0, 255, 0)
    
    # Process the video
    print(f"\nProcessing video: {input_path}")
    print(f"Output: {output_path} (Type: {output_type})")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    if downsample_ratio < 1.0:
        print(f"Using downsample ratio: {downsample_ratio} (faster but lower quality)")
    
    # Initialize recurrent states
    rec = None
    
    # Initialize progress bar
    progress = tqdm(total=total_frames, unit='frames')
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get background for this frame
        background = get_background(bg_type, (height, width), bg_color)
        
        # Process the frame
        output_frame, rec = process_frame(
            model=model,
            frame=frame_rgb,
            device=device,
            rec=rec,
            downsample_ratio=downsample_ratio,
            background=background
        )
        
        # Save the processed frame
        if output_type == 'frames':
            # Save as individual frame
            output_file = os.path.join(output_path, f"frame_{frame_idx:05d}.png")
            if output_frame.shape[2] == 4:
                # RGBA output
                cv2.imwrite(output_file, cv2.cvtColor(output_frame, cv2.COLOR_RGBA2BGRA))
            else:
                # RGB output
                cv2.imwrite(output_file, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        else:
            # Add to video
            if output_frame.shape[2] == 4:
                # For RGBA, remove alpha for video writer (only if not saving frames)
                output_frame_bgr = cv2.cvtColor(output_frame[:,:,0:3], cv2.COLOR_RGB2BGR)
                writer.write(output_frame_bgr)
            else:
                # For RGB
                output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                writer.write(output_frame_bgr)
        
        # Update progress
        frame_idx += 1
        progress.update(1)
    
    # Cleanup
    progress.close()
    cap.release()
    if writer:
        writer.release()
    
    print(f"\nVideo processing complete! Output saved to: {output_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Background Remover using RVM')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output video file or directory for frames')
    parser.add_argument('--model', '-m', type=str, default='models/rvm_resnet50.pth',
                       help='Path to RVM model file')
    parser.add_argument('--output-type', '-t', type=str, choices=['video', 'rgba', 'frames'],
                       default='video',
                       help='Output format: video (with background), rgba (transparent), or frames')
    parser.add_argument('--bg-color', type=int, nargs=3, default=[0, 255, 0],
                       help='Background color as R G B values (0-255)')
    parser.add_argument('--downsample', type=float, default=1.0,
                       help='Downsample ratio for faster processing (e.g., 0.5 for half resolution)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Computation device')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        output_type=args.output_type,
        bg_color=tuple(args.bg_color),
        downsample_ratio=args.downsample,
        device=args.device
    )

if __name__ == '__main__':
    main()