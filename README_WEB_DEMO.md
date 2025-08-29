# 多模态水印工具 Web Demo

## 🌟 项目简介

这是一个基于您现有多模态水印工具的Web演示应用，提供友好的浏览器界面来使用文本、图像、音频、视频四种模态的水印功能。

## ✨ 功能特性

### 🔧 核心功能
- **文本水印**: 基于CredID算法的文本生成与水印嵌入/提取
- **图像水印**: 基于VideoSeal/PRC算法的图像生成与水印处理
- **音频水印**: 基于AudioSeal算法的语音生成与音频水印
- **视频水印**: 基于HunyuanVideo的视频生成与VideoSeal水印

### 🌐 Web功能
- **直观界面**: 现代化的Web界面，支持拖拽上传
- **实时反馈**: 进度条显示和状态更新
- **结果下载**: 处理结果文件直接下载
- **错误处理**: 友好的错误提示和日志记录

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 已安装的多模态水印工具依赖
- 网络浏览器

### 1. 安装Web依赖

```bash
# 安装Web框架依赖
pip install -r requirements_web.txt

# 确保核心依赖已安装
pip install -r requirements.txt
```

### 2. 启动Web服务

#### 方法一: 使用启动脚本 (推荐)
```bash
python start_demo.py
```

#### 方法二: 直接运行
```bash
python app.py
```

### 3. 访问界面

启动成功后，在浏览器中访问：
- **本地访问**: http://localhost:5000
- **局域网访问**: http://你的IP地址:5000

## 📋 使用指南

### 1. 文本水印
**嵌入模式：**
1. 选择"文本"模态
2. 选择"嵌入水印"操作模式
3. 输入文本生成提示词（如："写一篇关于人工智能的文章"）
4. 输入要嵌入的水印消息
5. 点击"嵌入水印"按钮

**提取模式：**
1. 选择"文本"模态
2. 选择"提取水印"操作模式
3. 上传包含水印的文本文件（.txt, .doc, .docx等）
4. 点击"提取水印"按钮

### 2. 图像水印
**嵌入模式：**
1. 选择"图像"模态
2. 输入图像描述提示词（如："a beautiful cat under sunshine"）
3. 输入要嵌入的水印消息
4. 点击"嵌入水印"按钮生成带水印的图像

**提取模式：**
1. 选择"图像"模态
2. 选择"提取水印"操作模式
3. 上传图像文件（.jpg, .png, .gif等）
4. 点击"提取水印"按钮

### 3. 音频水印
**嵌入模式：**
1. 选择"音频"模态
2. 输入要转换为语音的文本内容
3. 输入要嵌入的水印消息
4. 点击"嵌入水印"按钮生成带水印的音频

**提取模式：**
1. 选择"音频"模态
2. 选择"提取水印"操作模式
3. 上传音频文件（.wav, .mp3, .flac等）
4. 点击"提取水印"按钮

### 4. 视频水印
**嵌入模式：**
1. 选择"视频"模态
2. 输入视频描述提示词（如："阳光洒在海面上"）
3. 输入要嵌入的水印消息
4. 点击"嵌入水印"按钮生成带水印的视频

**提取模式：**
1. 选择"视频"模态
2. 选择"提取水印"操作模式
3. 上传视频文件（.mp4, .avi, .mov等）
4. 点击"提取水印"按钮

## 🛠️ API 文档

### 基础接口

#### 1. 状态检查
```
GET /api/status
```
返回API状态和支持的算法信息。

#### 2. 水印嵌入
```
POST /api/embed
```
**参数:**
- `modality`: 模态类型 (text/image/audio/video)
- `prompt`: 提示词
- `message`: 水印消息
- 模态特定参数...

#### 3. 水印提取
```
POST /api/extract
```
**参数:**
- `modality`: 模态类型
- `file`: 上传的文件

#### 4. 结果下载
```
GET /api/download/<task_id>
```
下载处理结果文件。

### 请求示例

#### Python请求示例
```python
import requests

# 文本水印嵌入
response = requests.post('http://localhost:5000/api/embed', data={
    'modality': 'text',
    'prompt': '写一篇关于AI的文章',
    'message': 'demo_watermark'
})

# 图像水印提取
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/extract', 
                        data={'modality': 'image'}, files=files)
```

#### JavaScript请求示例
```javascript
// 水印嵌入
const formData = new FormData();
formData.append('modality', 'text');
formData.append('prompt', '写一篇关于AI的文章');
formData.append('message', 'demo_watermark');

fetch('/api/embed', {
    method: 'POST',
    body: formData
}).then(response => response.json());
```

## ⚙️ 配置说明

### 环境变量配置
复制 `.env.example` 到 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

主要配置项：
- `HOST`: 服务绑定地址 (默认: 0.0.0.0)
- `PORT`: 服务端口 (默认: 5000)
- `MAX_CONTENT_LENGTH`: 最大上传文件大小 (默认: 100MB)
- `HF_HOME`: HuggingFace缓存目录
- `DEVICE`: 计算设备 (cuda/cpu，留空自动检测)

### 文件大小限制
默认最大上传文件大小为100MB，可以通过环境变量 `MAX_CONTENT_LENGTH` 调整。

### 模型缓存
建议设置环境变量指向您的本地模型缓存：
```bash
export HF_HOME=/path/to/your/huggingface/cache
export TRANSFORMERS_CACHE=/path/to/your/transformers/cache
```

## 🔧 故障排除

### 常见问题

#### 1. 导入错误
**问题**: `ModuleNotFoundError: No module named 'src.unified.watermark_tool'`
**解决**: 确保在项目根目录运行，并且项目结构完整。

#### 2. 模型加载失败
**问题**: 水印工具初始化失败
**解决**: 
- 检查模型缓存是否存在
- 设置正确的 HuggingFace 缓存路径
- 确保网络连接正常（首次下载模型）

#### 3. 文件上传失败
**问题**: 文件上传后处理失败
**解决**:
- 检查文件格式是否支持
- 确认文件大小未超过限制
- 查看日志文件了解详细错误

#### 4. 内存不足
**问题**: CUDA out of memory 或系统内存不足
**解决**:
- 使用CPU模式：设置 `DEVICE=cpu`
- 减少批处理大小
- 关闭其他占用内存的程序

### 日志查看
查看详细日志信息：
```bash
tail -f watermark_demo.log
```

### 端口冲突
如果5000端口被占用，修改 `.env` 文件中的 `PORT` 设置。

## 📁 项目结构

```
unified_watermark_tool/
├── app.py                 # Flask主应用
├── start_demo.py         # 启动脚本
├── requirements_web.txt  # Web依赖
├── .env.example         # 环境配置示例
├── templates/           # HTML模板
│   └── index.html      # 主页面模板
├── demo_uploads/       # 上传文件目录
├── demo_outputs/       # 输出文件目录
├── src/                # 核心代码 (您的现有代码)
│   └── unified/
│       └── watermark_tool.py
└── README_WEB_DEMO.md  # 本文档
```

## 🤝 技术栈

- **后端**: Flask (Python Web框架)
- **前端**: HTML5 + CSS3 + JavaScript (无框架依赖)
- **文件处理**: Werkzeug
- **跨域**: Flask-CORS
- **核心算法**: 您的现有多模态水印工具

## 📝 开发说明

### 添加新功能
1. 在 `app.py` 中添加新的API路由
2. 在 `templates/index.html` 中添加对应的前端功能
3. 更新文档和测试

### 自定义界面
修改 `templates/index.html` 中的样式和布局，CSS变量定义在文件顶部方便自定义。

### 扩展API
在 `app.py` 中添加新的路由处理函数，遵循现有的错误处理和日志记录模式。

## 📄 许可证

本Web Demo继承原项目的许可证。

## 🆘 支持

如有问题，请检查：
1. 日志文件 `watermark_demo.log`
2. 控制台输出信息
3. 浏览器开发者工具的网络和控制台面板

---

**🎉 祝您使用愉快！**

如果遇到问题或需要功能改进，请随时反馈。