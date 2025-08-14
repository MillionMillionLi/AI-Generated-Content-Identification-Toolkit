# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install PyTorch dependencies first
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# For training (optional, may have installation issues)
pip install decord
```

#### 在现有 mmwt 环境最小安装（不升级 PyTorch 栈）
> 适用于已存在 Python 3.9 + torch 2.1.0 + torchvision 0.16.0 的环境，仅补齐缺失依赖并保持二进制栈稳定。

```bash
# 1) 切换到官方 PyPI（若内网镜像 SSL 异常）
pip install --no-input --no-color --index-url https://pypi.org/simple \
  ffmpeg-python av omegaconf timm==0.9.16 lpips pycocotools PyWavelets tensorboard calflops

# 2) 系统需有 ffmpeg 可执行：
ffmpeg -version  # 若无，请按系统方式安装（apt/yum/conda）

# 3) 运行流式推理（内置示例视频）
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/

# 运行成功后：
# - 输出视频：outputs/1.mp4
# - 嵌入的二进制消息：outputs/1.txt
# - 自动下载的模型权重：ckpts/y_256b_img.pth
```

> Python 3.9 兼容性（已在仓库中处理）：为兼容 `typing` 的联合类型写法（`int | float` 等），已在以下文件顶部加入 `from __future__ import annotations`：
> - `videoseal/videoseal/data/datasets.py`
> - `videoseal/videoseal/utils/cfg.py`
>
> 若你在其他分支或外部拷贝遇到相同错误（`TypeError: unsupported operand type(s) for |: 'type' and 'type'`），请同样添加上述语句或改用 `typing.Union`。

#### 常见网络/源问题（SSL 失败处理）
- 若出现 `TLS/SSL connection has been closed (EOF)` 或镜像不可用，改用官方源安装：
```bash
pip install --index-url https://pypi.org/simple <package>
```

### Code Quality Tools
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run tests with pytest
pytest

# Note: No specific test files found in repository - tests may be integrated in other ways
```

### Training Commands
```bash
# Image training (2 GPUs)
OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
    --video_dataset none --image_dataset sa-1b-full-resized --workers 8 \
    --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
    --epochs 601 --iter_per_epoch 1000

# Video finetuning (2 GPUs)
OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train.py --local_rank 0 \
    --video_dataset sa-v --image_dataset none --workers 0 --frames_per_clip 16 \
    --resume_from /path/to/ckpt/full/checkpoint.pth

# Single GPU training
torchrun train.py --debug_slurm

# Evaluation only
torchrun train.py --debug_slurm --only_eval True --output_dir output/
```

### Inference Commands
```bash
# Audio-visual watermarking
python inference_av.py --input assets/videos/1.mp4 --output_dir outputs/
python inference_av.py --detect --input outputs/1.mp4

# Streaming watermarking (for long videos)
python inference_streaming.py --input assets/videos/1.mp4 --output_dir outputs/

# Full evaluation of models
python -m videoseal.evals.full --checkpoint /path/to/videoseal/checkpoint.pth
python -m videoseal.evals.full --checkpoint baseline/wam
```

#### 运行结果说明（中文）
- 水印嵌入完成后会输出：
  - **水印视频**：`outputs/1.mp4`
  - **嵌入的二进制消息**：`outputs/1.txt`
  - **下载的模型权重**：`ckpts/y_256b_img.pth`（默认 256-bit 模型 `videoseal_1.0`）
- 提取流程会从 `outputs/1.mp4` 中恢复消息，并在日志显示 Bit Accuracy（例如 100.0%）。

## Architecture Overview

### Core Components

**VideoSeal Model Architecture:**
- **Embedder**: Embeds watermarks into videos/images (models: UNet, VAE-based)
- **Extractor/Detector**: Extracts and detects watermarks (models: ConvNeXT, DINO, Segmentation-based)
- **Augmenter**: Applies augmentations during training for robustness
- **JND (Just Noticeable Difference)**: Optional attenuation module for perceptual quality
- **WAM Base Class**: Shared functionality between watermarking models

**Model Types:**
- **videoseal_1.0**: 256-bit model (default, best balance)
- **videoseal_0.0**: 96-bit model (legacy, December 2024)

### Key Modules

- `videoseal/models/`: Core model implementations (Videoseal, embedders, extractors, baselines)
- `videoseal/modules/`: Building blocks (UNet, ConvNeXT, VAE, discriminator, etc.)
- `videoseal/losses/`: Loss functions (perceptual, SSIM, Watson, YUV, focal)
- `videoseal/data/`: Data loading and transforms
- `videoseal/augmentation/`: Augmentation strategies for robustness training
- `videoseal/evals/`: Evaluation scripts and metrics
- `videoseal/utils/`: Configuration management, logging, optimization utils

### Configuration System

Uses OmegaConf YAML-based configuration:
- `configs/embedder.yaml` / `configs/extractor.yaml`: Model architectures
- `configs/datasets/`: Dataset configuration files
- `configs/ablations/`: Experimental configurations
- Model cards in `videoseal/cards/`: Pre-trained model specifications

### Entry Points

- `train.py`: Main training script with distributed training support
- `inference_av.py`: Audio-visual watermarking
- `inference_streaming.py`: Memory-efficient streaming inference
- `videoseal.load()`: Programmatic model loading from model cards

### Video Processing

- **Chunk Processing**: Videos processed in chunks (default: 8 frames)
- **Step Size**: Watermark propagation across frames (default: 4 frames)
- **Temporal Consistency**: Maintains watermark consistency across video frames
- **Memory Efficiency**: Streaming processing for long videos

### Model Loading

```python
# Quick model loading (downloads automatically)
model = videoseal.load("videoseal")  # Default 256-bit model
model = videoseal.load("videoseal_0.0")  # Legacy 96-bit model

# Manual model loading with checkpoints
model = videoseal.load("path/to/checkpoint.pth")
```
- 用中文回答