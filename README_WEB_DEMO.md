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
2. 选择"嵌入水印"操作模式
3. 选择输入方式：
   - **AI生成内容**: 输入图像描述提示词（如："a beautiful cat under sunshine"）+ 水印消息
   - **上传现有文件**: 上传图像文件 + 水印消息
4. 点击"嵌入水印"按钮

**提取模式：**
1. 选择"图像"模态
2. 选择"提取水印"操作模式
3. 上传图像文件（.jpg, .png, .gif等）
4. 点击"提取水印"按钮

### 3. 音频水印
**嵌入模式：**
1. 选择"音频"模态
2. 选择"嵌入水印"操作模式
3. 选择输入方式：
   - **AI生成内容**: 输入要转换为语音的文本内容 + 水印消息
   - **上传现有文件**: 上传音频文件 + 水印消息
4. 点击"嵌入水印"按钮

**提取模式：**
1. 选择"音频"模态
2. 选择"提取水印"操作模式
3. 上传音频文件（.wav, .mp3, .flac等）
4. 点击"提取水印"按钮

### 4. 视频水印
**嵌入模式：**
1. 选择"视频"模态
2. 选择"嵌入水印"操作模式
3. 选择输入方式：
   - **AI生成内容**: 输入视频描述提示词（如："阳光洒在海面上"）+ 水印消息
   - **上传现有文件**: 上传视频文件 + 水印消息
4. 点击"嵌入水印"按钮

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
- `message`: 水印消息
- **生成模式参数:**
  - `prompt`: 提示词（用于AI生成内容）
  - 模态特定参数（如：resolution, num_frames等）
- **文件上传模式参数:**
  - `file`: 上传的文件（用于为现有文件添加水印）
  - `upload_mode`: 值为"true"，标识这是文件上传模式

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

## 🔄 前端界面重构说明

### 主要修改内容

#### 1. 用户交互流程优化
原有流程存在逻辑混乱问题，现已重新设计为清晰的分层选择：

**修改前的问题:**
- 点击"嵌入水印"后，选择输入方式会错误跳转到提取界面
- 操作模式选择器意外响应输入方式点击事件

**修改后的正确流程:**
```
1. 选择模态 (文本/图像/音频/视频)
   ↓
2. 选择操作模式 (嵌入水印/提取水印)
   ↓
3. [仅嵌入模式 & 非文本模态] 选择输入方式
   ├─ AI生成内容: 提示词输入 + 水印消息
   └─ 上传现有文件: 文件上传 + 水印消息
```

#### 2. HTML结构调整
- **输入方式选择**: 移至嵌入模式的最前面，逻辑更清晰
- **水印消息输入框**: 为生成模式和上传模式分别创建独立输入框
  - 生成模式: `message-input`
  - 上传模式: `message-input-upload`
- **显示控制**: 各个界面区域都有明确的显示/隐藏逻辑

#### 3. JavaScript事件处理修复
**核心问题修复:**
- `setupOperationModeSelector()` 使用通用选择器`.radio-card`导致误触发
- 修改为精确选择器：`document.querySelector('.operation-mode')`
- 添加安全检查：`if (!radio) return;`

**新增函数:**
- `getCurrentWatermarkMessage()`: 根据当前模式返回正确的水印消息
- 优化了各种模式切换的状态管理

#### 4. 前后端接口约定

**水印嵌入请求格式:**
```javascript
// AI生成模式
const formData = new FormData();
formData.append('modality', 'image/audio/video');
formData.append('prompt', '用户输入的提示词');
formData.append('message', '水印消息');
// + 模态特定参数

// 文件上传模式  
const formData = new FormData();
formData.append('modality', 'image/audio/video');
formData.append('file', uploadedFile);
formData.append('message', '水印消息'); 
formData.append('upload_mode', 'true'); // 标识上传模式
```

### 后端适配建议

#### 1. `/api/embed` 接口需要支持两种模式:

```python
@app.route('/api/embed', methods=['POST'])
def embed_watermark():
    modality = request.form.get('modality')
    message = request.form.get('message')
    
    if request.form.get('upload_mode') == 'true':
        # 文件上传模式 - 为现有文件添加水印
        uploaded_file = request.files.get('file')
        # 处理文件上传 + 水印嵌入
        result = process_file_with_watermark(uploaded_file, message, modality)
    else:
        # AI生成模式 - 生成内容 + 嵌入水印  
        prompt = request.form.get('prompt')
        # 处理内容生成 + 水印嵌入
        result = generate_content_with_watermark(prompt, message, modality)
    
    return jsonify(result)
```

#### 2. 参数获取逻辑:
- 检查 `upload_mode` 参数判断处理模式
- 生成模式: 使用 `prompt` 参数 
- 上传模式: 使用 `file` 参数
- 水印消息: 统一使用 `message` 参数

#### 3. 响应格式保持一致:
```json
{
  "task_id": "unique_id",
  "output_path": "path/to/result/file", 
  "generated_text": "生成的文本内容(仅文本模态)",
  "message": "嵌入的水印消息",
  "detected": true,
  "confidence": 0.95
}
```

### 测试验证
修复后的界面已通过以下场景测试:
- ✅ 文本模态：只显示生成模式，不显示输入方式选择
- ✅ 图像/音频/视频模态：正确显示输入方式选择
- ✅ AI生成内容：显示提示词+水印消息输入框
- ✅ 上传现有文件：显示文件上传+水印消息输入框  
- ✅ 提取模式：只显示文件上传，隐藏输入方式选择
- ✅ 模式切换：各种切换操作不会相互干扰

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