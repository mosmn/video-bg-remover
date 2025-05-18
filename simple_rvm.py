#!/usr/bin/env python3
"""
Simplified RVM video background removal for MacBook Air M1
Using the original RVM implementation
"""

import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import subprocess

# Add RVM repo to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rvm_repo'))

def init_model(model_path, device='mps'):
    """Initialize the RVM model"""
    from rvm_repo.model import MattingNetwork
    
    # Detect model type from filename
    if 'resnet50' in model_path.lower():
        model_type = 'resnet50'
    elif 'mobilenetv3_small' in model_path.lower():
        model_type = 'mobilenetv3_small'
    else:
        model_type = 'mobilenetv3'
    
    # Create model
    print(f"Creating {model_type} model...")
    model = MattingNetwork(model_type)
    
    # Load weights
    print(f"Loading model weights from {model_path}")
    if device == 'mps':
        # Apple Silicon needs special handling
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path, map_location=device)
        
    # Load state dict
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device
    if device == 'mps':
        device = torch.device('mps')
        print(f"Using MPS (Metal Performance Shaders) for Apple Silicon")
        model = model.eval().to(device)
    elif device == 'cuda' and torch.cuda.is_available():
        print(f"Using CUDA device")
        model = model.eval().cuda()
    else:
        print(f"Using CPU")
        model = model.eval().cpu()
        device = 'cpu'
    
    return model, device


def process_video(input_path, output_path, model_path, 
                  output_type='video', bg_color=(0, 255, 0), 
                  downsample_ratio=1.0, device='auto', transparent=False):
    """
    Process a video to remove the background
    
    Args:
        input_path: Path to the input video
        output_path: Path for the output video or folder
        model_path: Path to the RVM model weights (.pth file)
        output_type: Type of output ('video', 'frames', or 'transparent')
        bg_color: Background color (R, G, B)
        downsample_ratio: Downscale factor for faster processing
        device: Computing device ('auto', 'cpu', 'cuda', or 'mps')
        transparent: Whether to create a video with transparency
    """
    # Choose the best available device
    if device == 'auto':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'  # Apple Silicon
        elif torch.cuda.is_available():
            device = 'cuda'  # NVIDIA GPU
        else:
            device = 'cpu'   # CPU fallback
    
    # Load model
    model, device = init_model(model_path, device)
    
    # Check if input file exists
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found")
        return
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up output
    if output_type == 'frames':
        # Create frames directory
        os.makedirs(output_path, exist_ok=True)
        writer = None
    elif output_type == 'transparent':
        # Create temp frames directory for transparent output
        temp_frames_dir = f"{output_path}_temp_frames"
        os.makedirs(temp_frames_dir, exist_ok=True)
        writer = None
        transparent = True
    else:
        if transparent:
            # Create temp frames directory for transparent output
            temp_frames_dir = f"{output_path}_temp_frames"
            os.makedirs(temp_frames_dir, exist_ok=True)
            writer = None
        else:
            # Create video writer for standard output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"Error: Could not create output video: {output_path}")
                cap.release()
                return
    
    # Create normalized background color
    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:, :] = [c / 255.0 for c in bg_color]
    
    # Recurrent state
    r1 = r2 = r3 = r4 = None
    
    # Process frames
    print(f"Processing video: {os.path.basename(input_path)}")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Frames: {frame_count}")
    print(f"- Device: {device}")
    print(f"- Downsample ratio: {downsample_ratio}")
    
    # Progress bar
    with tqdm(total=frame_count, unit='frames') as progress:
        frame_idx = 0
        
        # Process frames
        with torch.no_grad():
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB and normalize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                
                # Convert to tensor
                tensor = torch.from_numpy(frame_rgb).float()
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Resize if needed
                if downsample_ratio != 1.0:
                    tensor_input = torch.nn.functional.interpolate(
                        tensor, 
                        scale_factor=downsample_ratio,
                        mode='bilinear', 
                        align_corners=False,
                        recompute_scale_factor=True
                    )
                else:
                    tensor_input = tensor
                
                # Move to device
                tensor_input = tensor_input.to(device)
                
                # Forward pass
                fgr, pha, *rec = model(tensor_input, r1, r2, r3, r4)
                r1, r2, r3, r4 = rec
                
                # Resize back to original resolution if needed
                if downsample_ratio != 1.0:
                    fgr = torch.nn.functional.interpolate(fgr, size=(height, width), mode='bilinear', align_corners=False)
                    pha = torch.nn.functional.interpolate(pha, size=(height, width), mode='bilinear', align_corners=False)
                
                # Convert to numpy
                fgr = fgr[0].cpu().numpy().transpose(1, 2, 0)
                pha = pha[0].cpu().numpy().transpose(1, 2, 0)
                
                # Process output based on type
                if transparent or output_type == 'transparent':
                    # RGBA output with transparency (PNG format)
                    rgba = np.concatenate([fgr, pha], axis=2)
                    rgba_uint8 = (rgba * 255).astype(np.uint8)
                    
                    # Save as PNG with transparency
                    if output_type == 'frames':
                        frame_path = os.path.join(output_path, f"frame_{frame_idx:05d}.png")
                    else:
                        frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:05d}.png")
                    
                    cv2.imwrite(frame_path, cv2.cvtColor(rgba_uint8, cv2.COLOR_RGBA2BGRA))
                
                elif output_type == 'frames':
                    # Composite foreground with background
                    composited = fgr * pha + bg * (1 - pha)
                    output_frame = (composited * 255).astype(np.uint8)
                    
                    # Save as image file
                    frame_path = os.path.join(output_path, f"frame_{frame_idx:05d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                
                else:
                    # Composite foreground with background
                    composited = fgr * pha + bg * (1 - pha)
                    output_frame = (composited * 255).astype(np.uint8)
                    
                    # Add to video
                    writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                
                # Update progress
                frame_idx += 1
                progress.update(1)
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    
    # If transparent video requested, create it from frames using ffmpeg
    if transparent or output_type == 'transparent':
        print("\nCreating transparent video from frames...")
        
        # Check if output format is specified
        _, ext = os.path.splitext(output_path)
        
        # Check if input video has audio
        has_audio = False
        audio_check = subprocess.run(
            ['ffmpeg', '-i', input_path, '-f', 'null', '-'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        if b'Audio:' in audio_check.stderr:
            has_audio = True
            print("Audio track detected in the input video - will preserve it")
        
        if ext.lower() == '.webm':
            # WebM with alpha channel
            if has_audio:
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), 
                    '-i', f'{temp_frames_dir}/frame_%05d.png',
                    '-i', input_path, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', 
                    '-lossless', '1', '-auto-alt-ref', '0',
                    '-c:a', 'libopus', '-b:a', '128k',
                    output_path
                ]
            else:
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), 
                    '-i', f'{temp_frames_dir}/frame_%05d.png',
                    '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', 
                    '-lossless', '1', '-auto-alt-ref', '0',
                    output_path
                ]
        elif ext.lower() == '.mov':
            # QuickTime with ProRes 4444 (supports alpha)
            if has_audio:
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), 
                    '-i', f'{temp_frames_dir}/frame_%05d.png',
                    '-i', input_path, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'prores_ks', '-profile:v', '4444',
                    '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16',
                    '-c:a', 'aac', '-b:a', '192k',
                    output_path
                ]
            else:
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), 
                    '-i', f'{temp_frames_dir}/frame_%05d.png',
                    '-c:v', 'prores_ks', '-profile:v', '4444',
                    '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16',
                    output_path
                ]
        else:
            # Default to QuickTime with ProRes 4444 if extension not recognized
            output_path_with_ext = os.path.splitext(output_path)[0] + '.mov'
            print(f"Warning: Output format not specified or doesn't support transparency. Using QuickTime with ProRes 4444: {output_path_with_ext}")
            
            if has_audio:
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), 
                    '-i', f'{temp_frames_dir}/frame_%05d.png',
                    '-i', input_path, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'prores_ks', '-profile:v', '4444',
                    '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16',
                    '-c:a', 'aac', '-b:a', '192k',
                    output_path_with_ext
                ]
            else:
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), 
                    '-i', f'{temp_frames_dir}/frame_%05d.png',
                    '-c:v', 'prores_ks', '-profile:v', '4444',
                    '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16',
                    output_path_with_ext
                ]
        
        # Run ffmpeg
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        # Clean up temp frames if needed
        if not output_type == 'frames':
            import shutil
            print(f"Cleaning up temporary frames in {temp_frames_dir}")
            shutil.rmtree(temp_frames_dir)
    
    print(f"Processing complete! Output saved to: {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Background Removal with RVM')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output video or directory for frames')
    parser.add_argument('--model', '-m', type=str, default='models/rvm_resnet50.pth',
                        help='Path to RVM model')
    parser.add_argument('--output-type', '-t', choices=['video', 'frames', 'transparent'], default='video',
                        help='Output format (video, frames, or transparent)')
    parser.add_argument('--bg-color', '-b', nargs=3, type=int, default=[0, 255, 0],
                        help='Background color as R G B values (0-255)')
    parser.add_argument('--downsample', '-d', type=float, default=1.0,
                        help='Downsample ratio for faster processing (e.g., 0.5 for half resolution)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                        help='Device for processing')
    parser.add_argument('--transparent', '-a', action='store_true',
                        help='Create video with transparency (requires ffmpeg)')
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Process video
    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        output_type=args.output_type,
        bg_color=tuple(args.bg_color),
        downsample_ratio=args.downsample,
        device=args.device,
        transparent=args.transparent
    )