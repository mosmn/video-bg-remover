#!/usr/bin/env python3
"""
RVM (Robust Video Matting) model implementation
Simplified from https://github.com/PeterL1n/RobustVideoMatting
"""

import torch
from torch import nn
from torch.nn import functional as F


class MattingNetwork(nn.Module):
    """
    Robust Video Matting network with recurrent architecture
    """
    
    def __init__(self, variant='mobilenetv3', pretrained_backbone=False):
        super().__init__()
        self.variant = variant
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3(pretrained_backbone)
            self.aspp = ASPP(in_channels=16, out_channels=16)
            self.decoder = RecurrentDecoder(in_channels=16, out_channels=4)
        elif variant == 'mobilenetv3_small':
            self.backbone = MobileNetV3Small(pretrained_backbone)
            self.aspp = ASPP(in_channels=16, out_channels=16)
            self.decoder = RecurrentDecoder(in_channels=16, out_channels=4)
        elif variant == 'resnet50':
            self.backbone = ResNet50(pretrained_backbone)
            self.aspp = ASPP(in_channels=2048, out_channels=256)
            self.decoder = RecurrentDecoder(in_channels=256, out_channels=32)
        else:
            raise NotImplementedError(f"Variant {variant} not implemented")
        
        # Project to output alpha and foreground
        self.project_mat = nn.Conv2d(32 if variant == 'resnet50' else 4, 1, kernel_size=1)
        self.project_seg = nn.Conv2d(32 if variant == 'resnet50' else 4, 4, kernel_size=1)
        
        # Additional processing for better matting quality
        self.refiner = Refiner()
        
    def forward(self, src, r1=None, r2=None, r3=None, r4=None):
        # Backbone
        f1, f2, f3, f4 = self.backbone(src)
        
        # ASPP
        f4 = self.aspp(f4)
        
        # Decoder
        h, r1, r2, r3, r4 = self.decoder(src, f1, f2, f3, f4, r1, r2, r3, r4)
        
        # Project to alpha and foreground
        alpha = self.project_mat(h)
        foreground = self.project_seg(h)
        
        # Refiner
        alpha = self.refiner(alpha, src)
        
        # Clamp alpha and apply to foreground
        alpha = torch.clamp(alpha, 0, 1)
        foreground = torch.clamp(foreground, 0, 1)
        
        return foreground, alpha, r1, r2, r3, r4


# Simplified implementation of backbone models
class MobileNetV3(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # Simplified placeholder - in the real implementation, this would be a proper MobileNetV3
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        
    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


class MobileNetV3Small(MobileNetV3):
    def __init__(self, pretrained=False):
        super().__init__(pretrained)


class ResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # Simplified placeholder - in the real implementation, this would be a proper ResNet50
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 256)
        self.layer2 = self._make_layer(256, 512)
        self.layer3 = self._make_layer(512, 1024)
        self.layer4 = self._make_layer(1024, 2048)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        out = aspp1 + aspp2
        return out


class RecurrentDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decode4 = RecurrentBlock(in_channels, out_channels * 2)
        self.decode3 = RecurrentBlock(out_channels * 2, out_channels * 2)
        self.decode2 = RecurrentBlock(out_channels * 2, out_channels * 2)
        self.decode1 = RecurrentBlock(out_channels * 2, out_channels)
        self.decode0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, f1, f2, f3, f4, r1=None, r2=None, r3=None, r4=None):
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, r3)
        x2, r2 = self.decode2(x3, r2)
        x1, r1 = self.decode1(x2, r1)
        x0 = self.decode0(x1)
        return x0, r1, r2, r3, r4


class RecurrentBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gru = GRU(out_channels, out_channels)
        
    def forward(self, x, recurrent=None):
        x = self.conv(x)
        x, h = self.gru(x, recurrent)
        return x, h


class GRU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ih = nn.Conv2d(in_channels, 3 * out_channels, kernel_size=3, padding=1)
        self.hh = nn.Conv2d(out_channels, 3 * out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, h=None):
        # Initial state if none provided
        if h is None:
            h = torch.zeros_like(x)
        
        # GRU computations
        ih = self.ih(x)
        hh = self.hh(h)
        
        z_ih, r_ih, n_ih = torch.chunk(ih, chunks=3, dim=1)
        z_hh, r_hh, n_hh = torch.chunk(hh, chunks=3, dim=1)
        
        z = torch.sigmoid(z_ih + z_hh)
        r = torch.sigmoid(r_ih + r_hh)
        n = torch.tanh(n_ih + r * n_hh)
        
        h = (1 - z) * n + z * h
        
        return h, h


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.box_filter.weight.data.fill_(1.0 / 9)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
    def forward(self, alpha, image):
        # Filter the alpha for guidance
        filtered_alpha = self.box_filter(alpha)
        
        # Concatenate with the input image
        x = torch.cat([image, alpha, filtered_alpha], dim=1)
        
        # Refine the alpha matte
        residual = self.conv(x)
        
        # Add the residual to get the final alpha
        refined_alpha = alpha + residual
        
        return refined_alpha