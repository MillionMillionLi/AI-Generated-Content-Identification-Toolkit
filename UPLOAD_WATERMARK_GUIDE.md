# 上传文件+水印嵌入功能使用指南

## 📋 功能概述

现在多模态水印工具Web Demo支持两种水印嵌入方式：

1. **AI生成内容+嵌入水印** (原有功能)
   - 用户输入提示词，AI生成内容并嵌入水印
   
2. **上传现有文件+添加水印** (新功能) ✨
   - 用户上传现有的图像/音频/视频文件，为其添加水印

## 🚀 新功能特性

### 支持的模态
- **图像水印**: 上传现有图像 (.jpg, .png, .gif等) → 添加VideoSeal水印
- **音频水印**: 上传现有音频 (.mp3, .wav, .flac等) → 添加AudioSeal水印  
- **视频水印**: 上传现有视频 (.mp4, .avi, .mov等) → 添加VideoSeal水印
- **文本水印**: 仅支持AI生成模式 (暂不支持上传文本文件)

### 技术实现
- 基于现有的UnifiedWatermarkEngine引擎
- 通过`image_input`、`audio_input`、`video_input`参数传递上传文件
- 无缝集成到现有的水印算法框架中

## 💻 Web界面使用方法

### 1. 选择操作模式
- 选择"嵌入水印"模式

### 2. 选择模态类型
- 点击选择：图像/音频/视频 (文本模态不支持上传)

### 3. 选择输入方式
- **AI生成内容**: 输入提示词让AI生成内容
- **上传现有文件**: 上传现有文件添加水印 ✨

### 4. 上传文件
- 点击上传区域或拖拽文件
- 支持的文件格式会根据选择的模态自动显示

### 5. 设置水印消息
- 输入要嵌入的水印信息

### 6. 执行嵌入
- 点击"嵌入水印"按钮开始处理

## 🔧 API接口使用方法

### 请求格式
```http
POST /api/embed
Content-Type: multipart/form-data

参数:
- modality: 模态类型 ('image', 'audio', 'video')
- message: 水印消息
- upload_mode: 'true' (标识为上传模式)
- file: 上传的文件
```

### 示例代码

#### Python示例
```python
import requests

# 图像水印嵌入
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'modality': 'image',
        'message': 'my_watermark',
        'upload_mode': 'true'
    }
    response = requests.post('http://localhost:5000/api/embed', 
                           files=files, data=data)
    result = response.json()
    print(f"任务ID: {result['task_id']}")
```

#### curl示例
```bash
# 音频水印嵌入
curl -X POST http://localhost:5000/api/embed \
  -F "modality=audio" \
  -F "message=my_audio_watermark" \
  -F "upload_mode=true" \
  -F "file=@audio.wav"
```

## 📊 技术架构

### 后端处理流程
```
1. 接收上传文件 → 保存到临时目录
2. 检测upload_mode参数 → 选择处理模式
3. 调用WatermarkTool.embed() → 传递xxx_input参数
4. UnifiedWatermarkEngine → 调用对应模态处理器
5. 保存水印后的文件 → 返回结果路径
```

### 关键代码修改
- **app.py**: `/api/embed`接口增加上传模式支持
- **unified_engine.py**: 已支持`image_input/audio_input/video_input`参数
- **index.html**: 前端界面已完整实现输入方式切换

## 🧪 测试方法

### 1. 启动服务器
```bash
python app.py
```

### 2. 运行测试脚本
```bash
python test_upload_watermark.py
```

### 3. 手动测试
- 打开浏览器访问 http://localhost:5000
- 选择图像/音频/视频模态
- 切换到"上传现有文件"模式
- 上传测试文件并嵌入水印

## 📁 文件类型支持

| 模态 | 支持格式 | 算法 |
|------|----------|------|
| 图像 | .jpg, .jpeg, .png, .gif, .bmp, .webp | VideoSeal/PRC |
| 音频 | .mp3, .wav, .flac, .aac, .ogg, .m4a | AudioSeal |
| 视频 | .mp4, .avi, .mov, .mkv, .flv, .webm | VideoSeal |

## ⚡ 性能说明

- **图像处理**: 通常1-10秒，取决于图像大小和算法
- **音频处理**: 通常2-30秒，取决于音频长度
- **视频处理**: 通常10秒-数分钟，取决于视频长度和分辨率

## 🛠️ 故障排除

### 常见问题
1. **文件上传失败**: 检查文件大小限制 (100MB) 和格式支持
2. **处理超时**: 大文件可能需要更长时间，请耐心等待
3. **水印嵌入失败**: 检查文件是否损坏，尝试其他格式

### 错误码说明
- `400`: 请求参数错误 (缺少文件或upload_mode参数)
- `413`: 文件太大 (超过100MB限制)
- `500`: 服务器内部错误 (水印嵌入失败)

## 🎯 未来改进

- [ ] 支持文本文件上传+水印嵌入
- [ ] 批量文件处理
- [ ] 更多文件格式支持
- [ ] 水印强度调节
- [ ] 进度实时反馈

---

**该功能基于统一的多模态水印引擎实现，与现有AI生成模式完全兼容。** 🎉