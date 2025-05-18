# Technical Documentation

This document provides technical details about the implementation of the Video Background Remover tool.

## Project Structure

- `simple_rvm.py` - Main script that handles video processing with transparency support
- `download_model.py` - Script to download the pre-trained RVM models
- `requirements.txt` - Python dependencies
- `models/` - Directory containing downloaded model weights
- `rvm_repo/` - Original RVM implementation that we integrate with

## Implementation Details

### RVM Model Integration

Our implementation leverages the Robust Video Matting (RVM) model from [the original repository](https://github.com/PeterL1n/RobustVideoMatting). We integrate with it in a way that enables acceleration on Apple Silicon via Metal Performance Shaders (MPS).

RVM uses a recurrent neural network approach to maintain temporal consistency, meaning each frame processing depends on the previous frames. The model produces two outputs:
- **Foreground prediction (FGR)**: The foreground RGB colors
- **Alpha matte (PHA)**: The opacity/transparency values

### Apple Silicon Optimization

The script automatically detects Apple Silicon hardware and uses PyTorch's MPS backend for acceleration. This results in significant performance improvements on M1/M2/M3 Macs compared to CPU processing:

```python
if device == 'auto':
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'  # Apple Silicon
    elif torch.cuda.is_available():
        device = 'cuda'  # NVIDIA GPU
    else:
        device = 'cpu'   # CPU fallback
```

### Video Processing Pipeline

The video processing pipeline consists of several key steps:

1. **Model Initialization**:
   - Load the appropriate RVM model based on the selected variant
   - Transfer model to the selected device (MPS, CUDA, or CPU)

2. **Frame-by-Frame Processing**:
   - Read each frame from the input video
   - Convert to RGB and normalize
   - Apply the RVM model to extract foreground and alpha matte
   - Track recurrent state between frames for temporal consistency

3. **Output Generation**:
   - For standard video: Composite foreground with colored background
   - For transparent output: Generate individual frames with alpha channel
   - For MP4 with alpha: Generate frames with green screen background

### Transparency Handling

Since different video formats handle transparency differently, we implement several approaches:

#### MOV (QuickTime)
For true transparency, we generate individual frames and use FFmpeg to create a ProRes 4444 video that preserves the alpha channel:

```python
cmd = [
    'ffmpeg', '-y', '-framerate', str(fps), 
    '-i', f'{temp_frames_dir}/frame_%05d.png',
    '-i', input_path, '-map', '0:v', '-map', '1:a',
    '-c:v', 'prores_ks', '-profile:v', '4444',
    '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16',
    '-c:a', 'aac', '-b:a', '192k',
    output_path
]
```

#### WebM
For web-compatible transparency, we use the VP9 codec with alpha support:

```python
cmd = [
    'ffmpeg', '-y', '-framerate', str(fps), 
    '-i', f'{temp_frames_dir}/frame_%05d.png',
    '-i', input_path, '-map', '0:v', '-map', '1:a',
    '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', 
    '-lossless', '1', '-auto-alt-ref', '0',
    '-c:a', 'libopus', '-b:a', '128k',
    output_path
]
```

#### MP4 with Green Screen
Since MP4 doesn't support alpha channels natively, we create a perfect green screen video that can be used with chroma keying:

```python
# Composite foreground with green background
composited = fgr * pha + green_bg / 255.0 * (1 - pha)
output_frame = (composited * 255).astype(np.uint8)
```

### Audio Preservation

We carefully preserve the original audio using FFmpeg's mapping capabilities. The process:

1. Detect if the input video contains an audio track
2. When creating output video, map the original audio to the new video
3. Use appropriate audio codec based on output format:
   - AAC for MOV (192kbps)
   - Opus for WebM (128kbps)

### Performance Optimization

The `--downsample` parameter allows users to trade quality for speed by processing at reduced resolution:

```python
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
```

After processing, the results are upscaled back to the original resolution.

## Customization and Extension

### Adding New Output Formats

To add support for a new output format, modify the `process_video` function in `simple_rvm.py`. You'll need to:

1. Detect the new file extension
2. Add appropriate FFmpeg command for that format
3. Configure both with-audio and without-audio variants

### Adding New Models

The current implementation supports three RVM model variants:
- MobileNetV3 Small
- MobileNetV3
- ResNet50

To add a new model:
1. Update the `download_model.py` script to include the new model option
2. Update the model detection in `init_model()` function in `simple_rvm.py`

## Dependencies

### Python Libraries

- **torch**: Neural network model backbone
- **torchvision**: Image processing utilities
- **opencv-python**: Video and image I/O operations
- **numpy**: Array operations
- **tqdm**: Progress bar visualization

### External Dependencies

- **FFmpeg**: Required for transparent video creation and audio processing

## Performance Considerations

- **Memory Usage**: The ResNet50 model uses significantly more memory than MobileNetV3
- **Processing Time**: Using `--downsample 0.5` provides a good balance between speed and quality
- **M1 Performance**: The MPS backend typically provides 3-5x speedup compared to CPU processing