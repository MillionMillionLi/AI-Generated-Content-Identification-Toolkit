# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified watermarking tool that supports both text and image watermarking. The project integrates multiple watermarking algorithms including:

- **Text Watermarking**: CredID algorithm for Large Language Model (LLM) identification
- **Image Watermarking**: PRC-Watermark and Stable Signature algorithms for image watermarking

## Architecture

The codebase follows a modular architecture with three main components:

### Core Modules
- `src/unified/watermark_tool.py`: Main unified interface providing both text and image watermarking capabilities
- `src/text_watermark/`: Text watermarking implementations, primarily CredID framework 
- `src/image_watermark/`: Image watermarking implementations including PRC-Watermark
- `src/utils/`: Shared utilities for configuration loading and model management

### CredID Text Watermarking Framework
Located in `src/text_watermark/credid/`, this is a comprehensive multi-party watermarking framework:

- `watermarking/`: Core watermarking algorithms (CredID, KGW, MPAC, etc.)
- `attacks/`: Attack implementations (copy-paste, deletion, homoglyph, substitution)
- `evaluation/`: Evaluation pipelines and metrics for quality, speed, robustness analysis
- `experiments/`: Experimental scripts for research validation
- `demo/`: Example scripts for single-party and multi-party scenarios

## Common Development Commands

### Installation and Setup
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install CredID-specific dependencies (if working with text watermarking)
pip install -r src/text_watermark/credid/watermarking/MPAC/requirements.txt

# Install PRC-Watermark dependencies (if working with image watermarking)  
pip install -r src/image_watermark/PRC-Watermark/requirements.txt
```

### Running the Tool
```bash
# Command line interface
watermark-tool --mode text --action embed --input "your text" --key "your_key"
watermark-tool --mode image --action embed --input "image.png" --key "your_key"

# Python interface (see examples/quick_start.py)
python examples/quick_start.py
```

### 使用视频水印（VideoSeal）最小封装

已将 VideoSeal 以子包形式集成于 `src/video_watermark/videoseal`，并在 `src/video_watermark/__init__.py` 暴露公共入口：

```python
from video_watermark import load

# 加载默认 256-bit 模型（会按需下载权重到运行目录 ckpts/）
model = load("videoseal")

# 对图像/视频执行嵌入或检测（参见 videoseal 文档）
# 例：对视频帧张量进行嵌入（FxCxHxW, 值域[0,1]）
# outputs = model.embed(frames, is_video=True)
# msgs = outputs["msgs"]
# frames_w = outputs["imgs_w"]
```

依赖提示：需要 `ffmpeg` 可执行和以下 Python 包（若缺请安装）
`ffmpeg-python av omegaconf timm==0.9.16 lpips pycocotools PyWavelets tensorboard calflops pytorch-msssim scikit-image scipy tqdm safetensors`

### Testing and Development
```bash
# PRC图像水印测试 (推荐)
python test_prc_only.py                 # 完整PRC水印系统测试
python test_modes_comparison.py         # 不同模式性能对比

# CredID文本水印测试
cd src/text_watermark/credid/demo
python test_method_single_party.py      # Single vendor scenario
python test_method_multi_party.py       # Multi-vendor scenario  
python test_real_word.py                # Real-world mixed text scenario

# Run specific experiments
cd src/text_watermark/credid/experiments
python run_CredID.py                    # CredID framework experiments
python run_attack.py                    # Attack robustness testing

# Run evaluation pipelines
cd src/text_watermark/credid/evaluation/pipelines
python success_rate_analysis.py         # Success rate evaluation
python quality_analysis.py              # Text quality analysis
python speed_analysis.py                # Performance analysis
```

### Configuration Management

The tool uses YAML configuration files:
- `config/default_config.yaml`: Main configuration for both text and image watermarking
- `config/text_config.yaml`: Text-specific configuration
- `src/text_watermark/credid/config/`: Algorithm-specific JSON configurations (CredID.json, KGW.json, etc.)

## Key Implementation Details

### Text Watermarking (CredID)
- Multi-bit watermarking framework supporting multiple LLM vendors
- Privacy-preserving design with Trusted Third Party (TTP) architecture
- Error correction codes (ECC) for robustness against attacks
- Joint-voting mechanism for multi-party watermark extraction

### Image Watermarking (PRC-Watermark)
- **完整的PRC水印系统**: 基于Stable Diffusion的伪随机纠错码水印
- **统一的exact_inversion实现**: 所有模式都使用相同的核心逆向函数，仅通过参数调节
- **多精度逆向模式**: 
  - FAST模式: 20步推理，decoder_inv=False，快速检测
  - ACCURATE模式: 50步推理，decoder_inv=True，精度平衡
  - EXACT模式: 50步推理，decoder_inv=True，最高精度
- **100%检测成功率**: 所有模式都能完美检测并解码水印消息
- **本地模型支持**: 离线模式使用本地Stable Diffusion 2.1模型
- **简洁架构**: 统一的`_image_to_latents()`函数，消除代码冗余

### Unified Interface
The `WatermarkTool` class in `src/unified/watermark_tool.py` provides:
- Consistent API for both text and image watermarking
- Batch processing capabilities
- Algorithm switching at runtime
- Configuration management across modalities

## Working with Different Components

When modifying text watermarking:
- Focus on `src/text_watermark/credid/` for core CredID implementation
- Use `src/text_watermark/credid/config/` for algorithm parameters
- Test changes with demo scripts in `src/text_watermark/credid/demo/`

When modifying image watermarking:
- **核心实现**: `src/image_watermark/prc_watermark.py` - PRC水印主要封装类
- **底层算法**: `src/image_watermark/PRC-Watermark/` - 原始PRC算法实现
- **高级接口**: `src/image_watermark/image_watermark.py` - 统一基类接口
- **测试方法**: 
  - 使用`python test_prc_only.py`进行完整系统测试
  - 测试所有三种模式(fast/accurate/exact)的性能表现

When extending the unified interface:
- Modify `src/unified/watermark_tool.py` for new functionality
- Update configuration schemas in `config/` directory
- Add examples to `examples/quick_start.py`

## PRC水印系统状态

### ✅ 已完成功能
- **核心架构**: 完整的PRCWatermark类封装，支持embed/extract统一接口
- **简洁实现**: 统一的`_image_to_latents()`函数，消除冗余代码，仅保留`exact_inversion()`
- **参数化控制**: 通过decoder_inv和inference_steps参数控制三种精度等级
- **100%检测成功**: 所有模式都能完美检测和解码水印消息
- **本地模型**: 离线模式支持，使用缓存的Stable Diffusion 2.1模型
- **完整测试**: 8项测试全部通过，代码简化后依然保持完美性能

### 🚀 性能基准
| 模式 | 检测成功率 | 处理时间 | 适用场景 |
|------|------------|----------|----------|
| FAST | 100% | 0.19秒 | 实时应用 |
| ACCURATE | 100% | 13.7秒 | 生产环境 |
| EXACT | 100% | 52.15秒 | 研究分析 |

### 🔧 技术实现亮点
- 解决了复杂的Python包导入冲突问题
- 实现了GPU/CPU tensor设备自动转换
- **代码架构优化**: 统一使用`exact_inversion()`函数，消除冗余的独立实现
- **参数化模式控制**: 通过decoder_inv和inference_steps参数实现不同精度等级
- 支持prompt引导的精确逆向(所有模式)

## Memory Annotations

- 用中文回答: 这是一个提醒，表示在处理项目或文档时使用中文进行交流和注释
- **PRC水印已完成**: 系统已经成功实现并通过所有测试，可以投入使用

## 变更摘要（2025-08）

### 背景
- 为兼容 Hunyuan 视频模型，环境升级至 `diffusers==0.34`。该版本与现有 PRC 图像水印路径存在不兼容风险（自定义管线模块注册差异）。

### 新增：VideoSeal 作为图像水印第二后端
- 在 `src/image_watermark/` 新增 `videoseal_image_watermark.py`，将单张图像视作单帧视频，复用 `src/video_watermark/videoseal_wrapper.py` 的 `embed/detect`。
- `src/image_watermark/image_watermark.py` 增加 `algorithm: videoseal` 分支，保持统一接口：
  - 直接对输入图像嵌入/提取
  - 或使用 Stable Diffusion 先生成图像，再用 VideoSeal 嵌入
- `src/unified/watermark_tool.py` 的 `get_supported_algorithms()['image']` 增加 `videoseal`。
- 检测增强：`VideoSealImageWatermark.extract(..., replicate=N, chunk_size=N)` 支持单图复制为多帧均值，提高读出稳定性与置信度。

### 懒加载与离线加载
- 懒加载：`ImageWatermark` 改为按需初始化具体后端，避免在构造时无关依赖（如 PRC/SD 管线）被加载。
- 离线加载（Stable Diffusion）：`src/utils/model_manager.py` 强制离线并解析本地 HF Hub 目录：
  - 优先解析 `.../huggingface/hub/models--stabilityai--stable-diffusion-2-1-base`（与 PRC 路径一致）
  - `from_pretrained(local_files_only=True)`，不触网

### 文本水印离线改造
- `test_complex_messages_real.py`：
  - 强制 `TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE`
  - `AutoTokenizer/AutoModelForCausalLM.from_pretrained(local_files_only=True, cache_dir=...)`
  - 自动探测本地缓存目录，或通过配置 `hf_cache_dir` 显式指定

### 导入与测试可用性
- 统一 `src.*` 绝对导入，确保以项目根运行脚本时稳定。
- `tests/conftest.py` 将 `src/` 注入 `sys.path`，保证测试环境下 `unified.*` 可导入。
- 新增单测与演示：
  - `tests/test_image_videoseal.py`（最小验证）
  - 根级 `test_image_videoseal_root.py`（可 `python` 直接运行）：
    - `--mode pil`：现有图像嵌入/提取
    - `--mode gen`：生成→嵌入→提取（完全离线，需本地 SD 权重）

### 使用指引（VideoSeal 图像水印）
- 配置（示例）：
```yaml
image_watermark:
  algorithm: videoseal
  model_name: stabilityai/stable-diffusion-2-1-base
  resolution: 512
  num_inference_steps: 30
  lowres_attenuation: true
  device: cuda
```
- 代码：
```python
from src.unified.watermark_tool import WatermarkTool
tool = WatermarkTool()
tool.set_algorithm('image', 'videoseal')
img = tool.generate_image_with_watermark(prompt='a cat', message='hello_videoseal')
res = tool.extract_image_watermark(img, replicate=16, chunk_size=16)
```
- 命令行演示：
```bash
python test_image_videoseal_root.py --mode pil  --device cuda
python test_image_videoseal_root.py --mode gen  --device cuda --resolution 512 --steps 30
```

### 提升检测置信度建议
- 生成侧：提高 `resolution` 与 `num_inference_steps`；简化 prompt；使用 GPU。
- 检测侧：`replicate` 设置为 8~32 并与 `chunk_size` 对齐，用多帧均值稳定读出。