# AI生成内容识别工具包

<div align="center">
  <!-- 项目logo占位符 - 需要logo图片 -->
  <!-- <a href="https://github.com/your-repo-link">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->


  <h3 align="center">AI生成内容标识技术开发套件</h3>

  <p align="center">
    一站式开源标识技术开发套件，支持文本、图像、音频和视频内容的显式标识，隐式标识和隐水印功能
    <br />
    <a href="#使用方法"><strong>快速开始 »</strong></a>
    <br />
    <br />
    <a href="pictures/watermark.mp4">在线演示</a>
    ·
    <a href="https://github.com/your-repo-link/issues">报告问题</a>
    ·
    <a href="https://github.com/your-repo-link/issues">请求功能</a>
  </p>

</div>

<!-- 目录 -->

<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于项目">关于项目</a>
      <ul>
        <li><a href="#构建技术">构建技术</a></li>
      </ul>
    </li>
    <li>
      <a href="#开始使用">开始使用</a>
      <ul>
        <li><a href="#前置要求">前置要求</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li><a href="#使用方法">使用方法</a></li>
    <li><a href="#网页演示">网页演示</a></li>
    <li><a href="#api参考">API参考</a></li>
    <li><a href="#性能">性能</a></li>
    <li><a href="#发展路线">发展路线</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>


## 关于项目

<!-- 项目截图占位符 - 需要网页界面截图 -->
<!-- [![产品截图][product-screenshot]](https://example.com) -->

本项目提供一站式开源标识技术开发套件。支持文本、图像、音频和视频四大模态，具备显式标识、隐式标识和隐水印功能，全面覆盖GB 45438-2025《网络安全技术 人工智能生成合成内容标识方法》"标准规定的标识范围。

### 为什么选择我们？

- **全面覆盖**：支持GB 45438-2025标准要求的所有标识方法
- **多模态支持**：统一处理文本、图像、音频和视频内容
- **双模式操作**：既支持AI内容生成，也支持现有文件处理
- **生产就绪**：配备完整的网页界面、批量处理和性能优化

### 构建技术

* [![Python][Python.org]][Python-url]
* [![PyTorch][PyTorch.org]][PyTorch-url]
* [![Flask][Flask.palletsprojects.com]][Flask-url]
* [![Transformers][Transformers-badge]][Transformers-url]
* [![Diffusers][Diffusers-badge]][Diffusers-url]

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开始使用

### 前置要求

- Python 3.8或更高版本
- 支持CUDA的GPU（推荐用于最佳性能）
- FFmpeg（视频处理必需）

### 安装

1. 克隆仓库

   ```bash
   git clone https://github.com/your-repo-link/unified_watermark_tool.git
   cd unified_watermark_tool
   ```

2. 安装核心依赖

   ```bash
   pip install -r requirements.txt
   ```

3. 安装模态特定依赖（可选）

   ```bash
   # 文本水印
   pip install -r src/text_watermark/credid/watermarking/MPAC/requirements.txt
   
   # 图像水印（PRC后端）
   pip install -r src/image_watermark/PRC-Watermark/requirements.txt
   
   # 音频水印
   pip install torch torchaudio julius soundfile librosa scipy matplotlib
   
   # 高级音频功能（Bark TTS）
   pip install git+https://github.com/suno-ai/bark.git
   ```

4. 配置环境（离线模式可选）

   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_HUB_OFFLINE=1
   export HF_ENDPOINT=https://hf-mirror.com  # 中国用户
   ```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用方法

### 快速开始

#### 水印

```python
from src.unified.watermark_tool import WatermarkTool

# 初始化工具
tool = WatermarkTool()

# 文本水印
watermarked_text = tool.embed("Please introduce AI generated content", "my_message", 'text')
result = tool.extract(watermarked_text, 'text')

# 图像水印
img = tool.embed("A cat under the sun", "img_watermark", 'image')

# 音频水印
audio = tool.embed("Hello world", "audio_watermark", 'audio',
                   output_path="output.wav")

# 视频水印
video = tool.embed("The sun shines on the sea", "video_watermark", 'video')
```

#### 显式标识

```python
from src.utils.visible_mark import (
    add_text_mark_to_text,
    add_overlay_to_image,
    add_overlay_to_video_ffmpeg,
    add_voice_mark_to_audio
)
from PIL import Image

# 文本显式标识
original_text = "这是一个示例文本内容。"
marked_text = add_text_mark_to_text(
    original_text,
    mark="本内容由人工智能生成",
    position="start"
)

# 图像显式标识
img = Image.open("input.jpg")
marked_img = add_overlay_to_image(
    img,
    "本内容由人工智能生成",
    position="bottom_right",
    font_percent=4.0,
    font_color="#FFFF00"
)

# 视频显式标识
marked_video_path = add_overlay_to_video_ffmpeg(
    "input.mp4",
    "output.mp4",
    "本内容由人工智能生成",
    position="bottom_right",
    font_percent=3.0,
    duration_seconds=3.0
)

# 音频显式标识（语音标注）
marked_audio_path = add_voice_mark_to_audio(
    "input.wav",
    "output.wav",
    "本内容由人工智能生成",
    position="start",
    voice_preset="v2/zh_speaker_6"
)
```

### 运行网页界面

```bash
# 启动Flask服务器
python app.py

# 打开浏览器并访问
# http://localhost:5000
```

### 高级配置

编辑`config/`目录中的配置文件：

- `config/default_config.yaml`：所有模态的全局设置
- `config/text_config.yaml`：文本水印特定设置

#### 配置示例

```yaml
# 文本水印配置
text_watermark:
  algorithm: "credid"                    # CredID算法用于LLM文本水印
  mode: "lm"                             # LM模式（语言模型，更高质量）
  credid:
    watermark_key: "default_key"          # 水印密钥标识符
    lm_params:
      delta: 1.5                         # 逻辑修改强度（水印强度）
      prefix_len: 10                     # 上下文分析的前缀长度
      message_len: 10                    # 水印消息长度（位）
    wm_params:
      encode_ratio: 8                    # 编码比率（每个水印位的令牌数）

# 图像水印配置
image_watermark:
  algorithm: "videoseal"                 # VideoSeal算法（默认）
  resolution: 512                       # AI生成模式的图像分辨率
  num_inference_steps: 30                # 推理步骤（越高质量越好）
  guidance_scale: 7.5                    # 引导比例（越高越符合提示）
  videoseal:
    replicate: 32                        # 多帧复制以增强检测
    chunk_size: 16                       # 处理块大小用于效率

# 音频水印配置
audio_watermark:
  algorithm: "audioseal"                 # AudioSeal音频水印算法
  sample_rate: 16000                     # AudioSeal所需采样率（16kHz）
  message_bits: 16                       # 水印消息长度（位）
  audioseal:
    nbits: 16                           # 消息编码位（基于SHA256）
    alpha: 1.0                          # 水印强度调整

# 视频水印配置
video_watermark:
  watermark: "videoseal"                 # VideoSeal水印算法
  videoseal:
    lowres_attenuation: true             # 启用低分辨率优化
    chunk_size: 16                       # 大视频的处理块大小
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 网页演示

![网页界面](pictures/web.png)

工具包包含一个综合的网页界面，具有以下特性：

### 🎨 界面特色

- **双栏布局**：左侧操作面板，右侧结果显示
- **模式切换**：在"AI生成"和"上传文件"模式之间切换
- **识别方法选择**：在"水印"和"显式标识"之间选择
- **实时对比**：原始内容与处理后内容的并排显示
- **多格式支持**：处理所有主要图像、音频和视频格式

### 🚀 支持的操作

| 模态     | 水印   | 显式标识   | 文件上传支持 |
| -------- | ------------ | ---------- | ------------ |
| **文本** | ✅ CredID     | ✅ 文本标注 | ❌ (仅生成)   |
| **图像** | ✅ VideoSeal  | ✅ 叠加标记 | ✅ 多格式     |
| **音频** | ✅ AudioSeal  | ✅ 语音标注 | ✅ 多格式     |
| **视频** | ✅  VideoSeal | ✅ 叠加标记 | ✅ 自动转码   |

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## API参考

### 核心API

#### 水印API

```python
# 嵌入水印
result = tool.embed(
    prompt="内容或提示",                    # 文本内容或生成提示
    message="watermark_message",          # 要嵌入的水印
    modality="text|image|audio|video",    # 内容类型
    **kwargs                              # 模态特定参数
)

# 提取水印
detection = tool.extract(
    content,                              # 带水印的内容
    modality="text|image|audio|video",    # 内容类型
    **kwargs                              # 检测参数
)
```

#### 显式标识API

```python
from src.utils.visible_mark import *

# 为不同模态添加显式标识
marked_text = add_text_mark_to_text(text, mark, position)
marked_image = add_overlay_to_image(image, text, position, font_percent)
marked_video_path = add_overlay_to_video_ffmpeg(input_path, output_path, text)
marked_audio_path = add_voice_mark_to_audio(input_path, output_path, mark_text)
```

### REST API端点

```bash
# 水印嵌入
POST /api/watermark/<modality>

# 显式标识
POST /api/visible_mark

# 任务状态检查
GET /api/status/<task_id>

# 文件服务
GET /api/files/<task_id>/original
GET /api/files/<task_id>/watermarked
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 发展路线

- [x] **水印实现**
  - [x] CredID文本水印
  - [x] VideoSeal图像/视频水印
  - [x] AudioSeal音频水印
  - [x] PRC-Watermark替代后端

- [x] **显式标识实现**
  - [x] 文本内容标注
  - [x] 图像叠加标记
  - [x] 视频叠加标记（FFmpeg）
  - [x] 音频语音标注（Bark TTS）

- [x] **网页界面**
  - [x] 双模式支持（AI生成+文件上传）
  - [x] 识别方法选择（不可见/可见）
  - [x] 实时对比显示
  - [x] 响应式设计
  - [x] 浏览器兼容的媒体转码

- [x] **合规与标准**
  - [x] GB 45438-2025合规
  - [x] 标准标记文本模板
  - [x] 可配置的定位和样式
  - [x] 多模态统一方法

- [ ] **未来增强**
  - [ ] 其他水印算法
  - [ ] 移动应用界面
  - [ ] 云部署选项
  - [ ] 高级分析仪表板
  - [ ] 界面多语言支持

查看[开放问题](https://github.com/your-repo-link/issues)获取完整的功能提议和已知问题列表。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 致谢

* [Meta AudioSeal](https://github.com/facebookresearch/audioseal) - 音频水印算法
* [VideoSeal](https://github.com/facebookresearch/videoseal) - 视频水印技术
* [Bark TTS](https://github.com/suno-ai/bark) - 文本转语音合成
* [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) - 文本转视频生成
* [Hugging Face](https://huggingface.co) - 模型托管和transformers库
* [PyTorch](https://pytorch.org) - 深度学习框架

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

<!-- MARKDOWN链接和图像 -->

[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Flask.palletsprojects.com]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[Transformers-badge]: https://img.shields.io/badge/🤗%20Transformers-FFD700?style=for-the-badge
[Transformers-url]: https://huggingface.co/transformers/
[Diffusers-badge]: https://img.shields.io/badge/🧨%20Diffusers-FF6B6B?style=for-the-badge
[Diffusers-url]: https://huggingface.co/docs/diffusers/