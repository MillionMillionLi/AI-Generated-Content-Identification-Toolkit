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

### Testing and Development
```bash
# Run CredID demos
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

### Image Watermarking
- Integration with Stable Diffusion models for generative watermarking
- PRC-Watermark implementation for robust image watermarking
- Support for both embedding watermarks in existing images and generating watermarked images

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
- Work with `src/image_watermark/PRC-Watermark/` for core implementation
- Use `src/image_watermark/image_watermark.py` for high-level interface

When extending the unified interface:
- Modify `src/unified/watermark_tool.py` for new functionality
- Update configuration schemas in `config/` directory
- Add examples to `examples/quick_start.py`

## Memory Annotations

- 用中文回答: 这是一个提醒，表示在处理项目或文档时使用中文进行交流和注释