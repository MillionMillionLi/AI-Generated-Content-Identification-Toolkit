# 🚀 多模态水印工具 Web Demo - 快速启动指南

## 📋 启动步骤

### 1. 安装依赖
```bash
# 安装Web依赖
pip install flask flask-cors

# 确保核心依赖已安装（如果还没有）
pip install torch transformers diffusers pillow numpy scipy
```

### 2. 启动服务
```bash
# 推荐方式：使用启动脚本
python start_demo.py

# 或者直接启动
python app.py
```

### 3. 访问界面
在浏览器中打开: **http://localhost:5000**

## 🎯 快速测试

### 文本水印测试
1. 选择"文本"模态
2. 输入提示词: "写一篇关于人工智能的简短介绍"
3. 水印消息: "test_demo_2025"
4. 点击"嵌入水印"

### 图像水印测试
1. 选择"图像"模态  
2. 输入提示词: "a beautiful cat under sunshine"
3. 水印消息: "image_demo"
4. 点击"嵌入水印"

## ⚠️ 常见问题

**问题**: 模块导入错误
**解决**: 确保在项目根目录运行

**问题**: 端口已占用
**解决**: 修改app.py中的端口或关闭占用5000端口的程序

**问题**: 模型加载失败  
**解决**: 确保网络连接正常，首次运行需要下载模型

## 🧪 运行测试
```bash
python test_web_demo.py
```

## 📁 生成的文件
- `demo_uploads/`: 用户上传的文件
- `demo_outputs/`: 处理结果文件  
- `watermark_demo.log`: 服务日志

---
**🎉 启动成功后就可以在浏览器中体验完整功能了！**