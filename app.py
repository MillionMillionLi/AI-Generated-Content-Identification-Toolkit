#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态水印工具 Web Demo - 主服务文件
Flask Web 应用，提供 REST API 接口
"""

import os
import sys
import uuid
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.unified.watermark_tool import WatermarkTool
    print("✅ 成功导入 WatermarkTool")
except ImportError as e:
    print(f"❌ 导入 WatermarkTool 失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'watermark_demo_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 文件大小限制

# 启用跨域支持
CORS(app)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watermark_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建必要目录
UPLOAD_FOLDER = project_root / 'demo_uploads'
OUTPUT_FOLDER = project_root / 'demo_outputs'
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# 支持的文件类型
ALLOWED_EXTENSIONS = {
    'text': {'.txt', '.doc', '.docx', '.pdf', '.md'},
    'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
    'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'},
    'video': {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
}

# 全局变量
watermark_tool = None
active_tasks = {}  # 存储活跃任务状态

def init_watermark_tool():
    """初始化水印工具"""
    global watermark_tool
    try:
        logger.info("正在初始化 WatermarkTool...")
        watermark_tool = WatermarkTool()
        logger.info("✅ WatermarkTool 初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ WatermarkTool 初始化失败: {e}")
        return False

def allowed_file(filename: str, modality: str) -> bool:
    """检查文件类型是否允许"""
    if not filename or '.' not in filename:
        return False
    
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS.get(modality, set())

def generate_task_id() -> str:
    """生成唯一任务ID"""
    return f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"

def save_uploaded_file(file, task_id: str, modality: str) -> str:
    """保存上传的文件"""
    if file and file.filename:
        filename = secure_filename(file.filename)
        # 添加时间戳和任务ID避免文件名冲突
        name, ext = os.path.splitext(filename)
        unique_filename = f"{task_id}_{name}{ext}"
        file_path = UPLOAD_FOLDER / unique_filename
        file.save(str(file_path))
        logger.info(f"文件已保存: {file_path}")
        return str(file_path)
    return ""

@app.route('/')
def index():
    """主页面 - 返回HTML界面"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"加载主页面失败: {e}")
        return jsonify({"error": f"加载页面失败: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """API状态检查"""
    try:
        # 检查工具是否初始化
        tool_status = "ready" if watermark_tool is not None else "not_initialized"
        
        # 获取支持的算法
        supported_algorithms = {}
        if watermark_tool:
            try:
                supported_algorithms = watermark_tool.get_supported_algorithms()
            except Exception as e:
                logger.warning(f"获取支持算法失败: {e}")
        
        return jsonify({
            "status": "online",
            "tool_status": tool_status,
            "supported_algorithms": supported_algorithms,
            "active_tasks": len(active_tasks),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"状态检查失败: {e}")
        return jsonify({"error": f"状态检查失败: {str(e)}"}), 500

@app.route('/api/embed', methods=['POST'])
def api_embed():
    """水印嵌入接口"""
    task_id = generate_task_id()
    active_tasks[task_id] = {"status": "processing", "progress": 0}
    
    try:
        # 检查工具是否可用
        if not watermark_tool:
            raise Exception("水印工具未初始化")
        
        # 获取请求数据
        modality = request.form.get('modality', 'text')
        prompt = request.form.get('prompt', '')
        message = request.form.get('message', 'demo_watermark')
        
        logger.info(f"[{task_id}] 开始嵌入水印 - 模态: {modality}")
        active_tasks[task_id]["progress"] = 20
        
        # 根据模态处理不同输入
        result = None
        output_path = None
        
        if modality == 'text':
            # 文本水印嵌入
            if not prompt.strip():
                raise ValueError("请提供文本生成提示词")
            
            active_tasks[task_id]["progress"] = 50
            result = watermark_tool.embed(prompt, message, 'text')
            
            # 保存生成的文本
            output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_text.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(result))
                
        elif modality == 'image':
            # 图像水印嵌入
            if not prompt.strip():
                raise ValueError("请提供图像描述提示词")
            
            active_tasks[task_id]["progress"] = 50
            # 获取额外参数
            resolution = int(request.form.get('resolution', 512))
            num_inference_steps = int(request.form.get('num_inference_steps', 30))
            
            result = watermark_tool.embed(prompt, message, 'image', 
                                        resolution=resolution,
                                        num_inference_steps=num_inference_steps)
            
            # 保存生成的图像
            output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_image.png"
            if hasattr(result, 'save'):  # PIL Image
                result.save(str(output_path))
            
        elif modality == 'audio':
            # 音频水印嵌入
            if not prompt.strip():
                raise ValueError("请提供音频内容提示词")
                
            active_tasks[task_id]["progress"] = 50
            
            # 获取额外参数
            voice_preset = request.form.get('voice_preset', 'v2/en_speaker_6')
            alpha = float(request.form.get('alpha', 1.0))
            
            output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_audio.wav"
            result = watermark_tool.embed(prompt, message, 'audio',
                                        voice_preset=voice_preset,
                                        alpha=alpha,
                                        output_path=str(output_path))
            
        elif modality == 'video':
            # 视频水印嵌入
            if not prompt.strip():
                raise ValueError("请提供视频描述提示词")
                
            active_tasks[task_id]["progress"] = 50
            
            # 获取额外参数
            num_frames = int(request.form.get('num_frames', 49))
            resolution = request.form.get('resolution', '512x768')
            width, height = map(int, resolution.split('x'))
            
            result = watermark_tool.embed(prompt, message, 'video',
                                        num_frames=num_frames,
                                        width=width,
                                        height=height)
            output_path = result  # 视频直接返回路径
            
        else:
            raise ValueError(f"不支持的模态类型: {modality}")
        
        # 更新任务状态
        active_tasks[task_id]["progress"] = 100
        active_tasks[task_id]["status"] = "completed"
        
        # 构建响应
        response_data = {
            "task_id": task_id,
            "status": "success",
            "modality": modality,
            "message": message,
            "output_path": str(output_path) if output_path else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加模态特定信息
        if modality == 'text':
            response_data["generated_text"] = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
        
        logger.info(f"[{task_id}] ✅ 嵌入完成")
        return jsonify(response_data)
        
    except Exception as e:
        # 更新任务状态为失败
        active_tasks[task_id]["status"] = "error"
        active_tasks[task_id]["error"] = str(e)
        
        logger.error(f"[{task_id}] ❌ 嵌入失败: {e}")
        return jsonify({
            "task_id": task_id,
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
        
    finally:
        # 清理任务记录 (5分钟后)
        if task_id in active_tasks:
            # 这里可以添加后台清理逻辑
            pass

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """水印提取接口"""
    task_id = generate_task_id()
    active_tasks[task_id] = {"status": "processing", "progress": 0}
    
    try:
        # 检查工具是否可用
        if not watermark_tool:
            raise Exception("水印工具未初始化")
        
        # 获取模态类型
        modality = request.form.get('modality', 'text')
        logger.info(f"[{task_id}] 开始提取水印 - 模态: {modality}")
        
        active_tasks[task_id]["progress"] = 20
        
        # 处理文件上传
        file = request.files.get('file')
        if not file or not file.filename:
            raise ValueError("请上传要检测的文件")
        
        # 检查文件类型
        if not allowed_file(file.filename, modality):
            raise ValueError(f"不支持的{modality}文件类型")
        
        # 保存上传文件
        file_path = save_uploaded_file(file, task_id, modality)
        active_tasks[task_id]["progress"] = 50
        
        # 执行水印提取
        if modality == 'text':
            # 读取文本文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            result = watermark_tool.extract(content, 'text')
            
        elif modality == 'image':
            # 图像水印提取
            result = watermark_tool.extract(file_path, 'image')
            
        elif modality == 'audio':
            # 音频水印提取  
            result = watermark_tool.extract(file_path, 'audio')
            
        elif modality == 'video':
            # 视频水印提取
            result = watermark_tool.extract(file_path, 'video')
            
        else:
            raise ValueError(f"不支持的模态类型: {modality}")
        
        # 更新任务状态
        active_tasks[task_id]["progress"] = 100
        active_tasks[task_id]["status"] = "completed"
        
        # 构建响应
        response_data = {
            "task_id": task_id,
            "status": "success",
            "modality": modality,
            "detected": result.get('detected', False),
            "message": result.get('message', ''),
            "confidence": result.get('confidence', 0.0),
            "metadata": result.get('metadata', {}),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[{task_id}] ✅ 提取完成")
        return jsonify(response_data)
        
    except Exception as e:
        # 更新任务状态为失败
        active_tasks[task_id]["status"] = "error"
        active_tasks[task_id]["error"] = str(e)
        
        logger.error(f"[{task_id}] ❌ 提取失败: {e}")
        return jsonify({
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/download/<task_id>')
def download_result(task_id):
    """下载处理结果"""
    try:
        # 查找输出文件
        output_files = list(OUTPUT_FOLDER.glob(f"{task_id}_*"))
        
        if not output_files:
            return jsonify({"error": "未找到结果文件"}), 404
        
        # 返回第一个匹配的文件
        file_path = output_files[0]
        return send_file(str(file_path), as_attachment=True)
        
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        return jsonify({"error": f"下载失败: {str(e)}"}), 500

@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    """获取任务状态"""
    if task_id in active_tasks:
        return jsonify(active_tasks[task_id])
    else:
        return jsonify({"error": "任务不存在"}), 404

if __name__ == '__main__':
    print("🚀 启动多模态水印工具 Web Demo")
    print("=" * 50)
    
    # 初始化水印工具
    if not init_watermark_tool():
        print("❌ 水印工具初始化失败，程序退出")
        sys.exit(1)
    
    print(f"📁 上传目录: {UPLOAD_FOLDER}")
    print(f"📁 输出目录: {OUTPUT_FOLDER}")
    print("🌐 Web 服务器启动中...")
    print("=" * 50)
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )