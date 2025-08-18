# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified watermarking tool that supports text, image, and audio watermarking. The project integrates multiple watermarking algorithms including:

- **Text Watermarking**: CredID algorithm for Large Language Model (LLM) identification
- **Image Watermarking**: PRC-Watermark and Stable Signature algorithms for image watermarking
- **Audio Watermarking**: AudioSeal algorithm for robust audio watermarking with Bark text-to-speech integration

## Architecture

The codebase follows a modular architecture with four main components:

### Core Modules
- `src/unified/watermark_tool.py`: Main unified interface providing text, image, and audio watermarking capabilities
- `src/text_watermark/`: Text watermarking implementations, primarily CredID framework 
- `src/image_watermark/`: Image watermarking implementations including PRC-Watermark
- `src/audio_watermark/`: Audio watermarking implementations using AudioSeal and Bark TTS
- `src/utils/`: Shared utilities for configuration loading and model management

### CredID Text Watermarking Framework
Located in `src/text_watermark/credid/`, this is a comprehensive multi-party watermarking framework:

- `watermarking/`: Core watermarking algorithms (CredID, KGW, MPAC, etc.)
- `attacks/`: Attack implementations (copy-paste, deletion, homoglyph, substitution)
- `evaluation/`: Evaluation pipelines and metrics for quality, speed, robustness analysis
- `experiments/`: Experimental scripts for research validation
- `demo/`: Example scripts for single-party and multi-party scenarios

### AudioSeal Audio Watermarking Framework
Located in `src/audio_watermark/`, this provides comprehensive audio watermarking capabilities:

- `audioseal_wrapper.py`: Core AudioSeal watermarking implementation with 16-bit message encoding/decoding and 3D tensor handling
- `bark_generator.py`: Bark text-to-speech integration with intelligent cache management and local model priority loading
- `audio_watermark.py`: Unified audio watermarking interface supporting both direct audio and TTS workflows with batch processing
- `utils.py`: Audio processing utilities for I/O, quality assessment, visualization, and noise robustness testing
- `audioseal/`: AudioSeal algorithm submodule (Meta's official implementation)

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

# Install AudioSeal dependencies (if working with audio watermarking)
pip install torch torchaudio julius soundfile librosa scipy matplotlib

# Install Bark for text-to-speech (optional, for advanced audio features)
pip install git+https://github.com/suno-ai/bark.git
```

### Running the Tool
```bash
# Command line interface
watermark-tool --mode text --action embed --input "your text" --key "your_key"
watermark-tool --mode image --action embed --input "image.png" --key "your_key"
watermark-tool --mode audio --action embed --input "audio.wav" --message "your_message"

# Python interface (see examples/quick_start.py)
python examples/quick_start.py

# Audio watermarking demo
python audio_watermark_demo.py
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

### 使用音频水印（AudioSeal）

已将 AudioSeal 集成于 `src/audio_watermark/`，提供完整的音频水印解决方案：

```python
from src.audio_watermark import create_audio_watermark

# 创建音频水印工具
watermark_tool = create_audio_watermark()

# 基础音频水印嵌入
import torch
audio = torch.randn(1, 16000)  # 1秒音频
message = "test_message_2025"

# 嵌入水印
watermarked_audio = watermark_tool.embed_watermark(audio, message)

# 提取水印
result = watermark_tool.extract_watermark(watermarked_audio)
print(f"检测成功: {result['detected']}, 消息: {result['message']}")

# 文本转语音 + 水印（需要安装Bark）
generated_audio = watermark_tool.generate_audio_with_watermark(
    prompt="Hello, this is a test",
    message="bark_watermark",
    voice_preset="v2/en_speaker_6"
)
```

**核心特性**：
- **16位消息编码**: 支持字符串消息的哈希编码
- **高质量嵌入**: SNR > 40dB，几乎无听觉差异
- **鲁棒检测**: 对噪声、压缩等攻击有良好抗性
- **多语言TTS**: 集成Bark支持中英文等多语言语音生成
- **批处理支持**: 支持批量音频处理
- **文件I/O**: 支持多种音频格式读写

**依赖要求**：
- 基础功能: `torch torchaudio julius soundfile librosa scipy matplotlib`
- 高级功能（TTS）: `pip install git+https://github.com/suno-ai/bark.git`
- 注意：Bark安装后会自动下载模型到指定缓存目录（约5GB）

### Testing and Development

#### 🚀 推荐的测试运行方式

我们提供了一个统一的测试脚本 `run_tests.py` 来简化测试流程，自动处理路径设置和依赖检查：

```bash
# 运行所有测试
python run_tests.py

# 运行特定模态的测试
python run_tests.py --audio             # 音频水印测试
python run_tests.py --image             # 图像水印测试  
python run_tests.py --text              # 文本水印测试

# 运行特定测试文件
python run_tests.py test_audio_watermark.py

# 详细输出
python run_tests.py --audio -v

# 快速测试（跳过耗时测试）
python run_tests.py --quick
```

#### 📁 传统的测试运行方式

如果你喜欢直接运行测试文件，我们已经修复了导入问题，以下命令都可以正常工作：

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

# AudioSeal音频水印测试
python tests/test_audio_watermark.py    # 完整音频水印测试套件
python audio_watermark_demo.py          # 端到端演示脚本
```

#### 🛠️ 故障排除

如果遇到导入问题：

1. **确保从项目根目录运行**：
   ```bash
   cd /path/to/unified_watermark_tool
   python run_tests.py --audio
   ```

2. **检查环境**：
   ```bash
   python -c "import torch; print('✅ PyTorch 可用')"
   python -c "import transformers; print('✅ Transformers 可用')"
   ```

3. **使用我们的便利脚本**：
   ```bash
   python run_tests.py --audio  # 自动设置路径和检查依赖
   ```

### Configuration Management

The tool uses YAML configuration files:
- `config/default_config.yaml`: Main configuration for text, image, and audio watermarking
- `config/text_config.yaml`: Text-specific configuration
- `src/text_watermark/credid/config/`: Algorithm-specific JSON configurations (CredID.json, KGW.json, etc.)

Audio watermarking configuration example:
```yaml
audio_watermark:
  algorithm: audioseal
  device: cuda
  nbits: 16
  sample_rate: 16000
  bark_config:
    model_size: large
    temperature: 0.8
    default_voice: v2/en_speaker_6
```

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

### Audio Watermarking (AudioSeal)
- **Meta AudioSeal算法**: 基于深度学习的鲁棒音频水印技术，完整Python封装
- **消息编码系统**: 16位消息支持，使用SHA256哈希确保编码一致性，支持字符串到二进制的可靠转换
- **高保真嵌入**: SNR>40dB（实测44.45dB），听觉质量几乎无损失
- **多模态集成**: 
  - 直接音频水印嵌入/提取（0.93秒嵌入，0.04秒提取）
  - Bark TTS集成实现文本→语音→水印的完整流程（3-8秒生成）
- **设备自适应**: 支持CPU/CUDA自动切换和设备张量管理，修复设备不匹配问题
- **批处理支持**: 高效的批量音频处理能力（3个音频2.8秒）
- **格式兼容**: 支持WAV、MP3、FLAC等主流音频格式读写
- **鲁棒性测试**: 对不同SNR级别噪声的抗性验证（SNR≥10dB可靠检测）

### Unified Interface
The `WatermarkTool` class in `src/unified/watermark_tool.py` provides:
- Consistent API for text, image, and audio watermarking
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

When modifying audio watermarking:
- **核心实现**: `src/audio_watermark/audioseal_wrapper.py` - AudioSeal水印封装类，处理3D张量维度要求
- **TTS集成**: `src/audio_watermark/bark_generator.py` - Bark文本转语音生成器，智能缓存管理
- **统一接口**: `src/audio_watermark/audio_watermark.py` - 音频水印统一基类，提供与图像、文本一致的API
- **工具函数**: `src/audio_watermark/utils.py` - 音频处理、质量评估、可视化、噪声鲁棒性测试
- **测试方法**:
  - 使用`python tests/test_audio_watermark.py`进行完整系统测试（100%检测成功率）
  - 使用`python audio_watermark_demo.py`查看端到端演示
- **关键修复**: 解决维度匹配、设备一致性、Bark导入检测等技术问题

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

## AudioSeal音频水印系统状态

### ✅ 已完成功能
- **完整AudioSeal集成**: Meta官方AudioSeal算法的完整Python封装
- **消息编码系统**: 基于SHA256哈希的16位消息编码，支持字符串到二进制的可靠转换
- **Bark TTS集成**: 完整的文本转语音功能，支持多语言和多音色
- **统一接口设计**: AudioWatermark基类提供与图像、文本水印一致的API
- **设备自适应**: 自动CPU/CUDA检测，内存优化和设备张量管理
- **批处理支持**: 高效的批量音频处理和水印操作
- **质量评估工具**: SNR、MSE、相关性等音频质量指标计算
- **多格式支持**: WAV、MP3、FLAC等音频格式的读写支持

### 🚀 性能基准
| 功能 | 处理时间 | 质量指标 | 检测成功率 |
|------|----------|----------|------------|
| 基础嵌入 | 0.93秒 | SNR: 44.45dB | 100% |
| 基础提取 | 0.04秒 | 相关性: >0.95 | 100% |
| TTS生成 | 3-8秒 | 自然度: 高 | 100% |
| 批处理(3个) | 2.8秒 | SNR: >40dB | 100% |

### 🔧 技术实现亮点
- **维度处理优化**: 解决了AudioSeal对3D张量(batch, channels, time)的严格要求，自动处理1D/2D输入
- **设备一致性**: 修复了CUDA/CPU张量设备不匹配的问题，确保所有张量在同一设备
- **消息匹配算法**: 通过原始消息列表匹配实现高准确率的消息解码
- **错误处理机制**: 完善的异常捕获和降级处理策略，支持自动回退
- **模型懒加载**: 按需加载AudioSeal生成器和检测器，优化内存使用
- **多语言支持**: Bark TTS支持中英文等多种语言的高质量语音合成
- **智能缓存管理**: Bark模型优先使用本地缓存，支持符号链接和自定义缓存目录
- **测试覆盖**: 包含基础功能、批处理、文件I/O、噪声鲁棒性等全面测试

## 🚨 已知问题与限制

### Bark TTS 缓存问题

**问题描述**:
- Bark TTS存在双重缓存系统问题，会同时使用HuggingFace缓存目录和专用的Suno缓存目录
- 即使设置了`HF_HOME`或`CACHE_DIR`，Bark仍会在`/root/.cache/suno/`下载约8.4GB的模型文件
- 这导致磁盘空间重复占用，特别是在存储空间有限的环境中

**根本原因**:
- Bark使用独立的模型管理系统，不完全遵循HuggingFace的缓存配置
- 存在两套缓存逻辑：HuggingFace标准缓存 + Suno专用缓存

**当前受限功能**:
- 文本转语音功能 (`generate_audio_with_watermark`)
- 高级音频水印演示 (`demo_text_to_audio_watermark`) 
- 完整模式演示 (`python audio_watermark_demo.py --mode full`)

**不受影响的功能**:
- 基础音频水印功能 (AudioSeal嵌入/提取)
- 基础模式演示 (`python audio_watermark_demo.py --mode basic`)
- 音频文件处理和质量评估
- 批处理功能

## Memory Annotations

- 用中文回答: 这是一个提醒，表示在处理项目或文档时使用中文进行交流和注释
- **PRC水印已完成**: 系统已经成功实现并通过所有测试，可以投入使用
- **AudioSeal音频水印已完成**: Meta AudioSeal算法完整集成，包含Bark TTS，全部测试通过，性能稳定

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

## HunyuanVideo 集成问题解决记录（2025-08）

### 背景
在集成 HunyuanVideo 文生视频模型时遇到了一系列技术问题，经过系统性排查和修复，现已完全解决。

### 遇到的主要问题及解决方案

#### 1. 网络连接和模型下载问题
**问题现象**：
- `Network is unreachable` 错误
- `Connection reset by peer` 连接重置
- 无法从 HuggingFace 官网和镜像站点下载模型

**根本原因**：
- 环境中设置了代理但可能不稳定
- 需要使用国内镜像站点 `hf-mirror.com`
- 原始 `tencent/HunyuanVideo` 仓库缺少 diffusers 兼容格式

**解决步骤**：
```bash
# 1. 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 2. 使用社区维护的 diffusers 兼容版本
hunyuanvideo-community/HunyuanVideo  # 而非 tencent/HunyuanVideo

# 3. 在代码中添加多重回退策略
repo_candidates = [
    "hunyuanvideo-community/HunyuanVideo",  # 优先：社区版本
    "tencent/HunyuanVideo"  # 回退：官方版本
]
```

#### 2. 内存管理策略冲突
**问题现象**：
```
It seems like you have activated a device mapping strategy on the pipeline so calling 'enable_model_cpu_offload()' isn't allowed.
```

**根本原因**：
同时使用了两种互斥的内存优化策略：
- `device_map="balanced"` (设备映射)
- `enable_model_cpu_offload()` (CPU卸载)

**解决方案**：
```python
# 智能内存优化策略选择
using_device_map = hasattr(self.pipeline, 'hf_device_map') and self.pipeline.hf_device_map is not None

if using_device_map:
    self.logger.info("检测到device_map，跳过enable_model_cpu_offload以避免冲突")
else:
    # 只有在没有使用device_map时才启用CPU offload
    if hasattr(self.pipeline, 'enable_model_cpu_offload'):
        self.pipeline.enable_model_cpu_offload()
```

#### 3. 数据类型兼容性问题
**问题现象**：
```
"replication_pad3d_cuda" not implemented for 'BFloat16'
```

**根本原因**：
`torch.bfloat16` 数据类型与某些CUDA操作不兼容

**解决方案**：
```python
# 使用 float16 替代 bfloat16
if self.device == 'cuda':
    torch_dtype = torch.float16  # 而非 torch.bfloat16
    device_map = "balanced"
else:
    torch_dtype = torch.float32
    device_map = None
```

#### 4. 输出格式处理问题
**问题现象**：
```
'list' object has no attribute 'shape'
```

**根本原因**：
HunyuanVideo 管道返回的是 PIL.Image 对象列表，而非期望的 numpy 数组

**解决方案**：
```python
# 处理不同的输出格式
if hasattr(result, 'frames') and result.frames is not None:
    video_frames = result.frames[0]  # 标准格式
elif isinstance(result, (list, tuple)) and len(result) > 0:
    video_frames = result[0]  # 直接返回列表

# 特别处理 PIL 图像列表
elif video_frames and hasattr(video_frames[0], 'convert'):
    from PIL import Image
    frames = []
    for img in video_frames:
        if isinstance(img, Image.Image):
            img_rgb = img.convert('RGB')
            frame_array = np.array(img_rgb)
            frames.append(frame_array)
    
    video_array = np.stack(frames, axis=0)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()
    video_tensor = video_tensor / 255.0
```

### 最终实现效果
**✅ 成功指标**：
- HunyuanVideo 模型完整下载（39GB，30个文件）
- 视频生成功能正常（13帧 320x320 分辨率）
- 管道加载时间：约50秒
- 视频生成时间：约3秒
- 输出格式：`torch.Size([13, 3, 320, 320])`

### 关键配置文件修改
1. **`src/video_watermark/model_manager.py`**：添加镜像支持和多仓库回退
2. **`src/video_watermark/hunyuan_video_generator.py`**：修复内存管理和数据类型问题
3. **环境变量**：设置 `HF_ENDPOINT=https://hf-mirror.com`

### 经验总结
1. **网络问题**：优先使用国内镜像，设置多重回退机制
2. **模型兼容性**：选择社区维护的 diffusers 格式版本
3. **内存优化**：避免同时使用冲突的优化策略
4. **数据类型**：在 CUDA 环境下使用 `float16` 而非 `bfloat16`
5. **输出处理**：支持多种输出格式，特别是 PIL 图像列表

### 测试命令
```bash
# 完整测试
export HF_ENDPOINT=https://hf-mirror.com
python tests/test_video_watermark_demo.py

# 预期输出：视频生成成功，保存到 tests/test_results/
```

## Video Watermarking (HunyuanVideo + VideoSeal) 快速参考

本项目提供基于 Diffusers 的 HunyuanVideo 文生视频与 VideoSeal 水印的一体化流程，采用本地快照离线加载，避免联网不确定性。模型用法参考其模型卡（Diffusers 示例）[HunyuanVideo 模型卡](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)。

### 代码结构（关键文件）
- `src/video_watermark/model_manager.py`
  - 管理 HunyuanVideo 本地缓存目录，优先使用本地快照，必要时可下载（测试默认关闭下载）。
- `src/video_watermark/hunyuan_video_generator.py`
  - 按工作脚本方式加载本地快照：
    - `HunyuanVideoTransformer3DModel.from_pretrained(local_path, subfolder="transformer", torch_dtype, local_files_only=True)`
    - `HunyuanVideoPipeline.from_pretrained(local_path, transformer=transformer, torch_dtype, local_files_only=True)`
    - CUDA 下启用 `vae.enable_tiling()` 与 `enable_model_cpu_offload()`，降低显存占用与黑屏风险。
  - 核心接口：
    - `generate_video(...) -> Union[list/np.ndarray, str]`（output_path 为 None 返回帧序列，否则保存文件）
    - `generate_video_tensor(...) -> torch.Tensor  # (frames, channels, H, W)`
- `src/video_watermark/videoseal_wrapper.py`
  - VideoSeal 嵌入与检测包装，包含字符串⇄bits 转换与分块检测处理。
- `src/video_watermark/utils.py`
  - 视频 I/O（OpenCV）、保存/读取、性能计时、GPU 内存监控等工具。
- `src/video_watermark/video_watermark.py`
  - 统一视频水印接口，对上层/统一引擎暴露：
    - `generate_video_with_watermark(prompt, message, ...) -> str`
    - `embed_watermark(video_path, message, ...) -> str`
    - `extract_watermark(video_path, max_frames=None, chunk_size=None) -> Dict`
    - `batch_process_videos(...) -> list`
    - `get_system_info() -> Dict` / `clear_cache()`

### 接口输入/输出（摘要）
- `HunyuanVideoGenerator.generate_video(prompt, negative_prompt=None, num_frames=49, height=720, width=1280, num_inference_steps=30, guidance_scale=6.0, seed=None, output_path=None)`
  - 输入：提示词、帧数（建议 4*k+1，如 13/49/75）、分辨率、步数等
  - 输出：帧序列或输出文件路径
- `HunyuanVideoGenerator.generate_video_tensor(...) -> torch.Tensor`
  - 输出：`(frames, channels, height, width)`，值域 `[0, 1]`
- `VideoWatermark.generate_video_with_watermark(prompt, message, ..., lowres_attenuation=True) -> str`
  - 输出：保存的视频路径
- `VideoWatermark.embed_watermark(video_path, message, ..., max_frames=None) -> str`
  - 输出：保存的视频路径
- `VideoWatermark.extract_watermark(video_path, max_frames=None, chunk_size=None) -> Dict[str, Any]`
  - 输出：`{"detected": bool, "message": str, "confidence": float, ...}`

### 典型用法
```python
from src.video_watermark.video_watermark import create_video_watermark

wm = create_video_watermark()

# 文生视频 + 水印（5秒@15fps → 75帧）
out_path = wm.generate_video_with_watermark(
    prompt="阳光洒在海面上",
    message="demo_msg",
    num_frames=75,
    height=320,
    width=512,
    num_inference_steps=30,
    seed=42
)

# 提取水印
result = wm.extract_watermark(out_path, max_frames=50)
```

### 测试与运行
- 回归测试脚本：`tests/test_video_watermark_demo.py`
  - 用例1：纯文生视频（包含非黑屏像素检查）
  - 用例2：文生视频 + 水印嵌入 + 提取
- 运行命令：
```bash
conda activate mmwt && python -u unified_watermark_tool/tests/test_video_watermark_demo.py
```

### 重要约定
- 仅离线加载本地 HunyuanVideo 快照（`local_files_only=True`），避免联网不确定性。
- CUDA 时启用 `vae.enable_tiling()` 与 `enable_model_cpu_offload()`；不与 `device_map` 混用。
- 5秒@15fps 可用 `num_frames=75`，分辨率如 `320x512`；OOM 时自动降参重试（生成器内置）。