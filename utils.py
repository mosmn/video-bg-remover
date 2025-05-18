#!/usr/bin/env python3
"""
Utility functions for video background removal using RVM.
"""

import os
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from typing import Tuple, Optional, Dict, Any

def load_model(model_path: str, device: str = 'auto') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the RVM model from the specified path.
    
    Args:
        model_path: Path to the model file
        device: Device to run the model on ('cpu', 'cuda', 'mps', or 'auto')
        
    Returns:
        model: The loaded model
        model_info: Dictionary with model information
    """
    # Choose the optimal device based on availability
    if device == 'auto':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Use Metal Performance Shaders for Apple Silicon
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    model_info = {}
    model_info['device'] = device
    
    # Determine model type from filename
    if 'resnet50' in model_path.lower():
        model_type = 'resnet50'
    elif 'mobilenetv3_small' in model_path.lower():
        model_type = 'mobilenetv3_small'
    else:
        model_type = 'mobilenetv3'
    
    model_info['model_type'] = model_type
    
    if device == 'mps':
        checkpoint = torch.load(model_path, map_location='cpu')
        model = MattingNetwork(model_type)
        # Some checkpoints may store the state dict directly or under 'model_state'
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        model = model.eval().to(torch.device(device))
    else:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model = MattingNetwork(model_type)
        # Some checkpoints may store the state dict directly or under 'model_state'
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        model = model.eval().to(device)
    
    return model, model_info

def process_frame(model: torch.nn.Module, 
                 frame: np.ndarray, 
                 device: str,
                 rec: Optional[Dict[str, torch.Tensor]] = None,
                 downsample_ratio: float = 1.0,
                 background: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
    """
    Process a single video frame with the RVM model.
    
    Args:
        model: The RVM model
        frame: Input video frame
        device: Device to run inference on
        rec: Recurrent state from previous frame
        downsample_ratio: Downsample ratio for faster processing
        background: Optional background image
        
    Returns:
        result: Processed frame with alpha
        rec: Updated recurrent state
    """
    # Convert frame to tensor and normalize
    frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255
    frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), 
                                scale_factor=downsample_ratio, 
                                mode='bilinear', 
                                align_corners=False, 
                                recompute_scale_factor=True)
    
    # Move tensor to device
    if device == 'mps':
        frame_tensor = frame_tensor.to(torch.device(device))
    else:
        frame_tensor = frame_tensor.to(device)
    
    # Process with model
    with torch.no_grad():
        if rec is None:
            fgr, pha, rec = model(frame_tensor, return_rvm_rec=True)
        else:
            fgr, pha, rec = model(frame_tensor, rec, return_rvm_rec=True)
    
    # Upscale output to original resolution
    fgr = F.interpolate(fgr, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
    pha = F.interpolate(pha, size=(frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
    
    # Convert tensors to numpy
    fgr = fgr[0].permute(1, 2, 0).cpu().numpy()
    pha = pha[0].permute(1, 2, 0).cpu().numpy()
    
    # Create output frame
    if background is not None:
        # Composite foreground with given background
        result = fgr * pha + background * (1 - pha)
        result = (result * 255).astype(np.uint8)
    else:
        # Create RGBA output
        result = np.concatenate([fgr * 255, pha * 255], axis=2).astype(np.uint8)
    
    return result, rec

def get_background(bg_type: str, frame_shape: Tuple[int, int], bg_color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Create a background based on specified type.
    
    Args:
        bg_type: Type of background ('color', 'image', or 'none')
        frame_shape: Shape of the video frame (height, width)
        bg_color: RGB color tuple if bg_type is 'color'
        
    Returns:
        background: Background image
    """
    if bg_type == 'color' and bg_color is not None:
        # Create solid color background
        bg = np.ones((frame_shape[0], frame_shape[1], 3), dtype=np.float32)
        bg[:, :, 0] = bg_color[0] / 255
        bg[:, :, 1] = bg_color[1] / 255
        bg[:, :, 2] = bg_color[2] / 255
        return bg
    elif bg_type == 'none':
        # Create transparent background for RGBA
        return None
    else:
        raise ValueError(f"Unsupported background type: {bg_type}")

# RVM Model definition - needed to load the model
class MattingNetwork(torch.nn.Module):
    """
    Base class for matting networks.
    
    Args:
        variant: Network variant.
            - 'mobilenetv3'
            - 'resnet50'
            - 'mobilenetv3_small'
    """
    def __init__(self, variant: str = 'mobilenetv3'):
        super().__init__()
        self.variant = variant
        
        # Creating a simplified backbone structure that matches the saved model's state_dict
        # This allows us to load the model weights correctly
        self.backbone = torch.nn.Module()
        self.backbone.encoder = torch.nn.Module()
        self.backbone.aspp = torch.nn.Module()
        self.backbone.decoder = torch.nn.Module()
        self.backbone.project_mat = torch.nn.Module()
        self.backbone.project_seg = torch.nn.Module()
        self.backbone.refiner = torch.nn.Module()
        
        print(f"Initialized MattingNetwork with variant: {variant}")
        
    def forward(self, src, r1=None, r2=None, r3=None, r4=None, return_rvm_rec=False):
        """
        Forward pass of the matting network.
        For inference only - this delegates to the backbone which will be loaded from the state_dict.
        """
        if isinstance(r1, tuple) and len(r1) == 4:  # Handle case where rec is passed as a tuple
            rec = r1
            fgr, pha, *new_rec = self.backbone(src, *rec)
        elif r1 is not None:  # Handle individual recurrent state components
            rec = (r1, r2, r3, r4)
            fgr, pha, *new_rec = self.backbone(src, *rec)
        else:  # Initial frame with no recurrent state
            fgr, pha, *new_rec = self.backbone(src)
        
        rec = tuple(new_rec)
        
        if return_rvm_rec:
            return fgr, pha, rec
        return fgr, pha