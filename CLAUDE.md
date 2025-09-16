# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a unified watermarking tool that supports text, image, audio, and video watermarking. The project integrates multiple watermarking algorithms including:

- **Text Watermarking**: CredID algorithm for Large Language Model (LLM) identification
- **Image Watermarking**: VideoSeal backend by default, PRC-Watermark optional
- **Audio Watermarking**: AudioSeal algorithm for robust audio watermarking with optional Bark text-to-speech integration
- **Video Watermarking**: HunyuanVideo generation + VideoSeal watermarking

## Architecture

The codebase follows a modular architecture with the following components:

### Core Modules
- `src/unified/unified_engine.py`: Core engine for multimodal watermarking (text/image/audio/video)
- `src/unified/watermark_tool.py`: High-level facade over the engine; preferred entry-point for users
- `src/text_watermark/`: CredID-based text watermarking implementation
- `src/image_watermark/`: Image watermarking (VideoSeal backend by default; PRC available)
- `src/audio_watermark/`: Audio watermarking (AudioSeal; optional Bark TTS)
- `src/video_watermark/`: Video generation (HunyuanVideo) + VideoSeal watermarking
- `src/utils/`: Shared utilities for configuration and model management

### Unified Engine (Updated)
Location: `src/unified/unified_engine.py`

Key features:
- Unified API: `embed(prompt, message, modality, **kwargs)` and `extract(content, modality, **kwargs)` for `text|image|audio|video`
- **Dual-mode support**: AI generation mode (prompt-based) and upload file mode (file-based watermarking)
- **Original file preservation**: returns both original and watermarked content for comparison display
- Defaults: `text=credid`, `image=videoseal`, `audio=audioseal`, `video=hunyuan+videoseal`
- Offline-first: lazily initializes text model/tokenizer from local cache; falls back to `sshleifer/tiny-gpt2` if configured model not found (still offline)
- Config-driven: reads `config/text_config.yaml` and modality-specific configs

Quick start:
```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()

# AI生成模式 (Generate Mode) - Web界面选择"AI生成内容"
# Text (文本水印仅支持AI生成模式)
txt = tool.embed("示例文本", "wm_msg", 'text')
res = tool.extract(txt, 'text')

# Image (图像AI生成 + 水印嵌入，自动保存原图和水印图用于对比)
img = tool.embed("a cat", "hello_vs", 'image')  # 返回水印图像

# Audio (音频AI生成 + 水印嵌入，自动保存原音频和水印音频)
aud = tool.embed("audio content", "hello_audio", 'audio', output_path="outputs/audio/a.wav")

# Video (视频AI生成 + 水印嵌入，自动保存原视频和水印视频)
vid = tool.embed("阳光洒在海面上", "video_wm", 'video')

# 上传现有文件模式 (Upload File Mode) - Web界面选择"上传现有文件"
# Image watermarking (图像文件水印嵌入)
img_wm = tool.embed("watermark message", "hello_img", 'image', 
                    image_input="/path/to/image.jpg")
                    
# Audio watermarking (音频文件水印嵌入)
aud_wm = tool.embed("watermark message", "hello_audio", 'audio',
                    audio_input="/path/to/audio.wav", output_path="outputs/watermarked.wav")
                    
# Video watermarking (视频文件水印嵌入，自动转码为浏览器兼容格式)
vid_wm = tool.embed("watermark message", "hello_video", 'video',
                    video_input="/path/to/video.mp4")
```

Parameters and returns:
- **Text**: 
  - **仅支持AI生成模式**，用于CredID文本水印生成
  - auto-uses cached model/tokenizer; returns watermarked `str`
  - extraction returns `{detected: bool, message: str, confidence: float}`
  
- **Image**: 
  - **AI生成模式**: 基于Stable Diffusion生成图像后嵌入水印，返回`PIL.Image`
  - **上传文件模式**: 使用`image_input`参数上传现有图像文件进行水印嵌入
  - **后端支持**: VideoSeal (默认), PRC-Watermark (可选)
  - **对比显示**: 自动保存原图和水印图，Web界面显示before/after对比
  - `extract` supports `replicate/chunk_size` for enhanced detection confidence
  
- **Audio**: 
  - **AI生成模式**: Bark TTS生成音频 + AudioSeal水印嵌入，`output_path`保存文件
  - **上传文件模式**: 使用`audio_input`参数上传现有音频文件进行水印嵌入  
  - **格式支持**: WAV, MP3, FLAC等主流音频格式
  - **对比显示**: 自动保存原音频和水印音频，支持Web播放器对比
  - returns `torch.Tensor | str`; extraction returns `{detected, message, confidence}`
  
- **Video**: 
  - **AI生成模式**: HunyuanVideo生成视频 + VideoSeal水印嵌入
  - **上传文件模式**: 使用`video_input`参数上传现有视频文件进行水印嵌入
  - **浏览器兼容**: 自动转码为H.264+AAC+faststart格式确保Web播放
  - **对比显示**: 自动保存原视频和水印视频，支持并排播放对比
  - returns saved video path; `extract` returns `{detected, message, confidence, metadata}`

Offline cache hints:
- Set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`; store models under `models/` or point `HF_HOME/HF_HUB_CACHE` to local hub

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
```python
from src.unified.watermark_tool import WatermarkTool

tool = WatermarkTool()
text_wm = tool.embed("这是一个测试文本", "msg", 'text')
text_res = tool.extract(text_wm, 'text')

# Optional: choose algorithms
tool.set_algorithm('image', 'videoseal')
img = tool.generate_image_with_watermark("a cat", message="hello")
img_res = tool.extract_image_watermark(img, replicate=16, chunk_size=16)
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

The tool uses YAML configuration files and supports both AI generation and file upload modes for all supported modalities:

#### 📁 主要配置文件位置
- `config/default_config.yaml`: 统一配置文件，包含所有模态的默认设置
- `config/text_config.yaml`: 文本水印专用配置
- `src/text_watermark/credid/config/`: 算法特定的JSON配置 (CredID.json, KGW.json, etc.)

#### 🔧 各模态参数配置详解

##### 文本水印配置 (Text Watermarking)
**修改文件**: `config/text_config.yaml` 或 `src/text_watermark/credid/config/CredID.json`
```yaml
# config/text_config.yaml
text_watermark:
  algorithm: credid
  model_name: gpt2-medium               # 或 sshleifer/tiny-gpt2 (离线回退)
  device: cuda                          # 设备选择：cpu/cuda
  watermark_method: credid
  hf_cache_dir: ~/.cache/huggingface   # 模型缓存目录
  offline_mode: true                   # 强制离线加载
```

**核心参数说明**:
- `model_name`: LLM模型路径，优先本地缓存
- `offline_mode`: 启用时强制`local_files_only=True`
- `watermark_method`: 支持credid, kgw, mpac等算法

##### 图像水印配置 (Image Watermarking)  
**修改文件**: `config/default_config.yaml` (image_watermark section)
```yaml
# config/default_config.yaml - 图像水印部分
image_watermark:
  algorithm: videoseal                  # 算法选择：videoseal（默认）, prc
  model_name: stabilityai/stable-diffusion-2-1-base
  resolution: 512                       # AI生成模式：图像分辨率
  num_inference_steps: 30               # AI生成模式：推理步数
  guidance_scale: 7.5                   # AI生成模式：引导系数
  lowres_attenuation: true              # VideoSeal：低分辨率衰减
  device: cuda
  
  # PRC-Watermark 特有配置（当algorithm=prc时生效）
  prc_config:
    decoder_model_path: models/dec_48b_whit.torchscript.pt
    noise_step: 50
    mode: exact                         # 模式选择：fast/accurate/exact
    
  # 上传文件模式配置
  upload_config:
    max_file_size: 10485760            # 最大上传文件大小 (10MB)
    supported_formats: [jpg, jpeg, png, bmp, webp]
```

**核心参数说明**:
- **AI生成模式**: `resolution`, `num_inference_steps`, `guidance_scale`控制生成质量
- **上传文件模式**: `max_file_size`和`supported_formats`控制文件上传限制
- **VideoSeal**: `lowres_attenuation`启用低分辨率优化，`replicate`和`chunk_size`提升检测精度
- **PRC**: `mode`选择检测精度级别，`noise_step`影响水印强度

##### 音频水印配置 (Audio Watermarking)
**修改文件**: `config/default_config.yaml` (audio_watermark section)
```yaml  
# config/default_config.yaml - 音频水印部分
audio_watermark:
  algorithm: audioseal                  # AudioSeal算法
  device: cuda
  nbits: 16                            # 消息编码位数
  sample_rate: 16000                   # 采样率
  
  # Bark TTS配置 (AI生成模式)
  bark_config:
    model_size: large                  # 模型大小：large/medium/small
    temperature: 0.8                   # 生成温度，控制随机性
    default_voice: v2/en_speaker_6     # 默认说话人音色
    cache_dir: ~/.cache/bark           # Bark模型缓存目录
    
  # 上传文件模式配置
  upload_config:
    max_file_size: 52428800           # 最大上传文件大小 (50MB)
    supported_formats: [wav, mp3, flac, aac, m4a]
    auto_resample: true               # 自动重采样到目标采样率
```

**核心参数说明**:
- **AI生成模式**: `bark_config`控制TTS质量，`temperature`影响语音自然度
- **上传文件模式**: `auto_resample`自动处理采样率不匹配问题  
- **AudioSeal**: `nbits=16`支持字符串消息编码，`sample_rate`需与输入音频匹配

##### 视频水印配置 (Video Watermarking)
**修改文件**: `config/default_config.yaml` (video_watermark section)
```yaml
# config/default_config.yaml - 视频水印部分  
video_watermark:
  # HunyuanVideo生成配置 (AI生成模式)
  hunyuan_config:
    model_name: hunyuanvideo-community/HunyuanVideo
    num_frames: 49                     # 视频帧数（建议4k+1：13,49,75）
    height: 720                        # 视频高度
    width: 1280                        # 视频宽度  
    num_inference_steps: 30            # 推理步数
    guidance_scale: 6.0                # 引导系数
    device: cuda
    enable_cpu_offload: true           # CPU内存卸载优化
    
  # VideoSeal水印配置  
  videoseal_config:
    model_path: ckpts/videoseal         # VideoSeal模型路径
    lowres_attenuation: true           # 低分辨率衰减
    device: cuda
    
  # 上传文件模式配置
  upload_config:
    max_file_size: 104857600          # 最大上传文件大小 (100MB)
    supported_formats: [mp4, avi, mov, mkv, flv, webm]
    transcode_for_web: true           # 自动转码为浏览器兼容格式
    target_codec: libx264             # 目标视频编解码器
    target_audio_codec: aac           # 目标音频编解码器
    enable_faststart: true            # 启用快速开始（Web优化）
```

**核心参数说明**:
- **AI生成模式**: `num_frames`控制视频长度，`height/width`控制分辨率，数值越高质量越好但耗时更长
- **上传文件模式**: `transcode_for_web=true`自动转码为H.264+AAC+faststart确保浏览器兼容
- **内存优化**: `enable_cpu_offload`在GPU内存不足时启用CPU卸载
- **质量平衡**: 降低分辨率和帧数可减少内存占用和处理时间

#### 🎯 快速参数调优指南

**提升生成质量**:
```yaml
# 图像：提高分辨率和推理步数
resolution: 1024
num_inference_steps: 50

# 视频：提高分辨率和帧数  
height: 1024
width: 1024
num_frames: 75

# 音频：使用更大的Bark模型
bark_config:
  model_size: large
  temperature: 0.7
```

**优化性能和内存**:
```yaml
# 降低分辨率和步数
resolution: 320
num_inference_steps: 20
height: 320
width: 512

# 启用内存优化
enable_cpu_offload: true
lowres_attenuation: true
```

**增强检测准确率**:
```yaml
# 图像VideoSeal检测优化
replicate: 16          # 单图复制为多帧 
chunk_size: 16         # 分块检测

# PRC精确模式
prc_config:
  mode: exact          # 最高精度检测
  noise_step: 50       # 标准噪声步数
```

## Web Demo Interface Features

### 🌐 统一Web界面 (templates/index.html)
项目提供了完整的Web演示界面，支持所有模态的水印操作：

**核心功能**:
- **双模式切换**: "AI生成内容" 和 "上传现有文件" 模式无缝切换
- **实时状态反馈**: 任务状态实时更新，支持进度显示和错误提示  
- **对比显示界面**: 自动显示原文件vs水印文件的并排对比
- **多媒体支持**: 支持文本、图像、音频、视频的Web播放和显示
- **文件下载**: 支持原文件和水印文件的独立下载

**界面特性**:
```javascript
// 动态模板切换
function toggleMode(modality) {
    // 根据选择的模态和模式显示相应的输入界面
    showTemplate(modality, selectedMode);
}

// 结果对比显示
function showComparison(modality, originalUrl, watermarkedUrl) {
    // 自动加载对比模板并显示before/after效果
    loadComparisonTemplate(modality, originalUrl, watermarkedUrl);
}
```

**技术实现亮点**:
- **媒体加载重试**: `loadMediaWithRetry()`确保音视频文件稳定加载
- **浏览器兼容优化**: 视频自动转码为H.264+AAC格式确保跨浏览器支持
- **响应式设计**: 支持桌面和移动设备的最佳显示效果

### 🔧 后端API支持 (app.py)
Flask后端提供完整的RESTful API：

**主要端点**:
```python
@app.route('/api/watermark/<modality>', methods=['POST'])
def watermark_endpoint(modality):
    # 统一水印处理端点，支持text/image/audio/video
    
@app.route('/api/files/<task_id>/original')
@app.route('/api/files/<task_id>/watermarked') 
def serve_file(task_id, file_type):
    # 文件服务端点，支持原文件和水印文件访问
    
@app.route('/api/status/<task_id>')
def get_task_status(task_id):
    # 任务状态查询，实时返回处理进度
```

**文件处理特性**:
- **智能文件管理**: 自动保存原文件和水印文件，支持task_id关联
- **多格式支持**: 自动处理不同文件格式的上传和转换  
- **内存优化**: 大文件流式处理，避免内存溢出
- **安全验证**: 文件类型验证和大小限制确保系统安全

## Key Implementation Details

### Text Watermarking (CredID)
- **仅支持AI生成模式**: 基于LLM的文本生成与水印嵌入
- Multi-bit watermarking framework supporting multiple LLM vendors
- Privacy-preserving design with Trusted Third Party (TTP) architecture
- Error correction codes (ECC) for robustness against attacks
- Joint-voting mechanism for multi-party watermark extraction
- **离线优先**: 优先使用本地缓存模型，支持完全离线运行

### Image Watermarking (Dual Backend Support)
**支持后端**: VideoSeal (默认), PRC-Watermark (可选)
**双模式支持**: AI生成模式 + 上传文件模式

#### VideoSeal 图像水印 (默认)
- **单帧视频处理**: 将图像视作单帧视频，复用VideoSeal视频水印算法
- **检测增强**: 支持`replicate`和`chunk_size`参数，通过多帧复制提升检测置信度
- **低分辨率优化**: `lowres_attenuation`参数优化低分辨率图像处理
- **AI生成模式**: Stable Diffusion 2.1 + VideoSeal水印，自动保存原图和水印图
- **上传文件模式**: 直接对上传图像嵌入VideoSeal水印，支持多种图像格式

#### PRC-Watermark (可选后端)
- **完整的PRC水印系统**: 基于Stable Diffusion的伪随机纠错码水印
- **统一的exact_inversion实现**: 所有模式都使用相同的核心逆向函数，仅通过参数调节
- **多精度逆向模式**: 
  - FAST模式: 20步推理，decoder_inv=False，快速检测
  - ACCURATE模式: 50步推理，decoder_inv=True，精度平衡
  - EXACT模式: 50步推理，decoder_inv=True，最高精度
- **100%检测成功率**: 所有模式都能完美检测并解码水印消息
- **本地模型支持**: 离线模式使用本地Stable Diffusion 2.1模型
- **简洁架构**: 统一的`_image_to_latents()`函数，消除代码冗余

#### 技术实现特点
- **懒加载架构**: 按需初始化具体后端，避免无关依赖加载
- **离线优先**: 强制本地模型加载，避免网络依赖
- **对比显示**: Web界面自动显示原图vs水印图的并排对比
- **格式支持**: JPG, PNG, BMP, WebP等主流图像格式

### Audio Watermarking (AudioSeal)
**双模式支持**: AI生成模式(Bark TTS) + 上传文件模式
**核心算法**: Meta AudioSeal 深度学习音频水印

#### 技术特性
- **16位消息编码**: 基于SHA256哈希的字符串消息编码系统，确保编码一致性
- **高保真嵌入**: SNR>40dB（实测44.45dB），听觉质量几乎无损失
- **双处理模式**: 
  - **AI生成模式**: Bark TTS文本转语音 + AudioSeal水印嵌入（3-8秒生成）
  - **上传文件模式**: 直接对上传音频文件进行水印嵌入（0.93秒嵌入，0.04秒提取）
- **原文件保存**: 两种模式都自动保存原音频和水印音频，支持Web界面对比播放
- **设备自适应**: 支持CPU/CUDA自动切换和设备张量管理，修复设备不匹配问题
- **批处理支持**: 高效的批量音频处理能力（3个音频2.8秒）
- **格式兼容**: 支持WAV、MP3、FLAC、AAC、M4A等主流音频格式读写
- **鲁棒性验证**: 对不同SNR级别噪声的抗性测试（SNR≥10dB可靠检测）

#### Bark TTS集成
- **多语言支持**: 支持中英文等多语言高质量语音合成
- **音色选择**: 支持多种预设说话人音色，默认使用`v2/en_speaker_6`
- **智能缓存**: 优先使用本地缓存，支持符号链接和自定义缓存目录
- **参数控制**: `temperature`控制生成随机性，`model_size`控制模型质量

### Video Watermarking (HunyuanVideo + VideoSeal)
**双模式支持**: AI生成模式(HunyuanVideo) + 上传文件模式
**技术栈**: HunyuanVideo文生视频 + VideoSeal视频水印

#### 核心特性
- **双处理流程**:
  - **AI生成模式**: HunyuanVideo文本生成视频 + VideoSeal水印嵌入
  - **上传文件模式**: 直接对上传视频文件进行VideoSeal水印嵌入
- **原文件保存**: 两种模式都自动保存原视频和水印视频，支持Web界面并排播放对比
- **浏览器兼容**: 自动转码为H.264+AAC+faststart格式，确保跨浏览器Web播放
- **内存优化**: 支持CPU内存卸载和VAE tiling，处理大分辨率视频
- **离线优先**: 优先使用本地HunyuanVideo模型快照，避免网络下载

#### HunyuanVideo集成
- **模型支持**: 使用`hunyuanvideo-community/HunyuanVideo`社区维护版本
- **帧数控制**: 支持13/49/75等帧数配置（建议4k+1格式）
- **分辨率配置**: 支持320x320到1280x720等多种分辨率
- **内存管理**: CUDA下启用`vae.enable_tiling()`和`enable_model_cpu_offload()`
- **生成时间**: 典型13帧320x320视频约3秒生成时间

#### VideoSeal视频水印
- **256位水印**: 支持长消息的字符串编码
- **分块检测**: 支持`max_frames`和`chunk_size`参数优化大视频处理
- **置信度评估**: 提供检测置信度和元数据信息
- **格式支持**: MP4, AVI, MOV, MKV, FLV, WebM等视频格式

#### 浏览器兼容转码
- **自动转码**: `transcode_for_web=true`启用H.264+AAC+faststart转码
- **编码器选择**: 默认使用libx264视频编码器和AAC音频编码器  
- **快速开始**: 启用faststart优化Web流媒体播放
- **文件管理**: 智能处理转码后的文件命名和访问

### Unified Interface (Enhanced)
The `WatermarkTool` class in `src/unified/watermark_tool.py` provides:
- **Consistent API**: 统一的`embed()`和`extract()`接口支持所有模态
- **Dual-mode support**: 每个模态都支持AI生成和文件上传两种模式
- **Original file preservation**: 自动保存原文件和水印文件供对比显示
- **Batch processing**: 批处理能力支持大规模文件处理
- **Algorithm switching**: 运行时算法切换（如VideoSeal/PRC图像后端）
- **Configuration management**: 跨模态配置管理和参数优化
- **Web integration**: 与Flask Web界面的无缝集成
- **Error handling**: 完善的错误处理和降级策略

## Working with Different Components

### 🔧 各模态开发和修改指南

#### 修改文本水印 (Text Watermarking)
**主要文件位置**:
- **核心算法**: `src/text_watermark/credid/` - CredID多方水印框架
- **配置文件**: `src/text_watermark/credid/config/` - 算法特定参数（CredID.json等）
- **统一配置**: `config/text_config.yaml` - 全局文本水印设置
- **测试脚本**: `src/text_watermark/credid/demo/` - 单方和多方场景演示

**修改流程**:
1. 算法参数调整 → 修改`config/text_config.yaml`或相应JSON配置文件
2. 模型路径设置 → 配置`model_name`和`hf_cache_dir`参数
3. 离线模式 → 设置`offline_mode: true`和相应环境变量
4. 测试验证 → 运行`src/text_watermark/credid/demo/`下的测试脚本

#### 修改图像水印 (Image Watermarking)  
**主要文件位置**:
- **统一接口**: `src/image_watermark/image_watermark.py` - 双后端支持的基类
- **VideoSeal后端**: `src/image_watermark/videoseal_image_watermark.py` - 默认后端实现
- **PRC后端**: `src/image_watermark/prc_watermark.py` - 可选PRC水印实现
- **配置文件**: `config/default_config.yaml` (image_watermark section)

**修改流程**:
1. **切换后端** → 修改`algorithm: videoseal|prc`配置
2. **AI生成参数** → 调整`resolution`, `num_inference_steps`, `guidance_scale`
3. **上传文件限制** → 修改`upload_config`中的`max_file_size`和`supported_formats`
4. **检测优化** → 配置VideoSeal的`replicate`和`chunk_size`参数
5. **测试验证** → 运行`python test_image_videoseal_root.py`或`python test_prc_only.py`

#### 修改音频水印 (Audio Watermarking)
**主要文件位置**:
- **统一接口**: `src/audio_watermark/audio_watermark.py` - 双模式音频水印基类
- **AudioSeal核心**: `src/audio_watermark/audioseal_wrapper.py` - 深度学习水印实现
- **Bark TTS**: `src/audio_watermark/bark_generator.py` - AI语音生成集成
- **工具函数**: `src/audio_watermark/utils.py` - 音频处理和质量评估
- **配置文件**: `config/default_config.yaml` (audio_watermark section)

**修改流程**:
1. **基础参数** → 调整`nbits`, `sample_rate`, `device`配置
2. **TTS设置** → 修改`bark_config`中的`model_size`, `temperature`, `default_voice`
3. **上传支持** → 配置`upload_config`的格式支持和文件大小限制  
4. **设备优化** → 根据硬件配置选择CPU/CUDA设备
5. **测试验证** → 运行`python tests/test_audio_watermark.py`完整测试套件

#### 修改视频水印 (Video Watermarking)
**主要文件位置**:
- **统一接口**: `src/video_watermark/video_watermark.py` - 双模式视频水印
- **HunyuanVideo**: `src/video_watermark/hunyuan_video_generator.py` - AI视频生成
- **VideoSeal**: `src/video_watermark/videoseal_wrapper.py` - 视频水印算法
- **视频处理**: `src/video_watermark/utils.py` - 转码和I/O工具
- **配置文件**: `config/default_config.yaml` (video_watermark section)

**修改流程**:
1. **生成质量** → 调整`num_frames`, `height`, `width`, `num_inference_steps`
2. **内存优化** → 配置`enable_cpu_offload`和设备映射策略  
3. **浏览器兼容** → 设置`transcode_for_web`, `target_codec`等转码参数
4. **上传支持** → 修改`upload_config`的视频格式和大小限制
5. **测试验证** → 运行`python tests/test_video_watermark_demo.py`

#### 扩展统一接口 (Unified Interface)
**主要文件位置**:
- **核心工具**: `src/unified/watermark_tool.py` - 高层API封装
- **引擎核心**: `src/unified/unified_engine.py` - 底层执行引擎
- **Web集成**: `app.py` - Flask Web应用后端
- **前端界面**: `templates/index.html` - 用户交互界面

**扩展流程**:
1. **新功能接口** → 在`watermark_tool.py`中添加新方法
2. **引擎支持** → 在`unified_engine.py`中实现底层逻辑
3. **配置更新** → 修改`config/`目录下的YAML配置文件
4. **Web集成** → 更新`app.py`的API端点和`templates/index.html`的界面
5. **测试覆盖** → 添加相应的测试用例和演示脚本

### 🎯 常见开发场景

**添加新的水印算法**:
1. 在相应模态目录下创建新的算法实现类
2. 更新模态基类的算法选择逻辑  
3. 添加配置项到YAML配置文件
4. 实现测试用例验证功能

**优化Web界面体验**:
1. 修改`templates/index.html`的前端逻辑
2. 更新`app.py`的API端点支持
3. 增强错误处理和用户反馈
4. 测试跨浏览器兼容性

**提升处理性能**:
1. 调整模型加载和推理参数
2. 优化内存使用和设备分配
3. 实现批处理和并行处理
4. 性能基准测试和监控

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