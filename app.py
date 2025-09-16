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
import math
import logging
import threading
import hashlib
import shutil
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from flask import Flask, request, jsonify, send_file, render_template, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.unified.watermark_tool import WatermarkTool
    from src.video_watermark.utils import VideoTranscoder
    print("✅ 成功导入 WatermarkTool")
except ImportError as e:
    print(f"❌ 导入 WatermarkTool 失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

class CandidateMessageManager:
    """
    候选消息管理器
    
    负责保存和管理水印嵌入时的原始消息和编码后的二进制数据，
    用于提取水印时进行智能匹配，提高提取准确率
    """
    
    def __init__(self, file_path: str = "candidate_messages.json"):
        """
        初始化候选消息管理器
        
        Args:
            file_path: 候选消息存储文件路径
        """
        self.file_path = Path(file_path)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 确保存储文件存在
        self._ensure_file_exists()
        
        self.logger.info(f"候选消息管理器初始化完成，存储文件: {self.file_path}")
    
    def _ensure_file_exists(self):
        """确保候选消息存储文件存在"""
        if not self.file_path.exists():
            with self.lock:
                if not self.file_path.exists():  # 双重检查
                    try:
                        with open(self.file_path, 'w', encoding='utf-8') as f:
                            json.dump({}, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"创建候选消息存储文件: {self.file_path}")
                    except Exception as e:
                        self.logger.error(f"创建候选消息存储文件失败: {e}")
                        raise
    
    def _load_candidates(self) -> Dict[str, Any]:
        """从文件加载候选消息"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"加载候选消息失败: {e}，返回空字典")
            return {}
        except Exception as e:
            self.logger.error(f"读取候选消息文件失败: {e}")
            return {}
    
    def _save_candidates(self, candidates: Dict[str, Any]):
        """保存候选消息到文件"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(candidates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存候选消息失败: {e}")
            raise
    
    def save_message(self, original_message: str, encoded_binary: List[int], 
                    task_id: str, modality: str = "text") -> str:
        """
        保存原始消息和编码后的二进制数据
        
        Args:
            original_message: 原始水印消息
            encoded_binary: 编码后的二进制数字列表
            task_id: 任务ID
            modality: 模态类型
            
        Returns:
            消息的唯一标识符
        """
        with self.lock:
            try:
                # 生成消息的唯一标识符
                message_content = f"{original_message}_{encoded_binary}_{modality}"
                message_id = hashlib.md5(message_content.encode()).hexdigest()[:16]
                
                # 加载现有候选消息
                candidates = self._load_candidates()
                
                # 保存新消息
                candidates[message_id] = {
                    "original_message": original_message,
                    "encoded_binary": encoded_binary,
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "modality": modality
                }
                
                # 保存到文件
                self._save_candidates(candidates)
                
                self.logger.info(f"保存候选消息: {message_id} -> '{original_message}' ({len(encoded_binary)} segments)")
                return message_id
                
            except Exception as e:
                self.logger.error(f"保存候选消息失败: {e}")
                raise
    
    def get_candidates(self, modality: str = "text") -> Dict[str, Dict[str, Any]]:
        """
        获取指定模态的所有候选消息
        
        Args:
            modality: 模态类型
            
        Returns:
            候选消息字典
        """
        with self.lock:
            candidates = self._load_candidates()
            # 过滤指定模态的消息
            filtered = {k: v for k, v in candidates.items() 
                       if v.get('modality', 'text') == modality}
            return filtered
    
    def find_best_match(self, decoded_binary: List[int], 
                       threshold: float = 0.4, modality: str = "text") -> Tuple[Optional[str], float]:
        """
        查找与解码二进制最匹配的候选消息
        
        Args:
            decoded_binary: 解码出的二进制数字列表
            threshold: 匹配阈值（0.0-1.0）
            modality: 模态类型
            
        Returns:
            (最佳匹配的原始消息, 匹配度分数) 或 (None, 0.0)
        """
        if not decoded_binary:
            return None, 0.0
        
        with self.lock:
            candidates = self.get_candidates(modality)
            
            best_match = None
            best_score = 0.0
            
            for message_id, candidate_data in candidates.items():
                candidate_binary = candidate_data.get('encoded_binary', [])
                if not candidate_binary:
                    continue
                
                # 计算匹配度
                score = self._calculate_match_score(decoded_binary, candidate_binary)
                
                if score > best_score and score >= threshold:
                    best_match = candidate_data['original_message']
                    best_score = score
                    
                    self.logger.debug(f"找到更好匹配: '{best_match}' (score: {score:.3f})")
            
            if best_match:
                self.logger.info(f"最佳匹配: '{best_match}' (score: {best_score:.3f}, threshold: {threshold})")
            else:
                self.logger.info(f"未找到满足阈值的匹配 (threshold: {threshold})")
                
            return best_match, best_score
    
    def _calculate_match_score(self, decoded: List[int], candidate: List[int]) -> float:
        """
        计算两个二进制序列的匹配度
        
        Args:
            decoded: 解码的二进制序列
            candidate: 候选的二进制序列
            
        Returns:
            匹配度分数 (0.0-1.0)
        """
        if not decoded or not candidate:
            return 0.0
        
        # 完全匹配
        if decoded == candidate:
            return 1.0
        
        # 长度匹配 + 部分匹配
        if len(decoded) == len(candidate):
            matches = sum(1 for a, b in zip(decoded, candidate) if a == b)
            return matches / len(decoded)
        
        # 前缀匹配（处理截断情况）
        min_len = min(len(decoded), len(candidate))
        if min_len > 0:
            prefix_matches = sum(1 for a, b in zip(decoded[:min_len], candidate[:min_len]) if a == b)
            max_len = max(len(decoded), len(candidate))
            
            # 前缀匹配得分 = (实际匹配数 / 最大长度) * 前缀权重
            return (prefix_matches / max_len) * 0.8
        
        return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取候选消息统计信息"""
        with self.lock:
            candidates = self._load_candidates()
            
            stats = {
                'total_messages': len(candidates),
                'by_modality': {},
                'recent_messages': 0  # 最近24小时内的消息
            }
            
            # 按模态统计
            for candidate in candidates.values():
                modality = candidate.get('modality', 'unknown')
                stats['by_modality'][modality] = stats['by_modality'].get(modality, 0) + 1
                
                # 统计最近消息
                try:
                    timestamp = datetime.fromisoformat(candidate.get('timestamp', ''))
                    if (datetime.now() - timestamp).total_seconds() < 86400:  # 24小时
                        stats['recent_messages'] += 1
                except:
                    pass
            
            return stats
    
    def clear_old_messages(self, days: int = 30):
        """清理超过指定天数的旧消息"""
        with self.lock:
            candidates = self._load_candidates()
            cutoff_time = datetime.now().timestamp() - (days * 86400)
            
            # 过滤掉旧消息
            filtered_candidates = {}
            removed_count = 0
            
            for message_id, candidate in candidates.items():
                try:
                    timestamp = datetime.fromisoformat(candidate.get('timestamp', ''))
                    if timestamp.timestamp() > cutoff_time:
                        filtered_candidates[message_id] = candidate
                    else:
                        removed_count += 1
                except:
                    # 时间戳解析失败，保留消息
                    filtered_candidates[message_id] = candidate
            
            # 如果有消息被移除，保存更新
            if removed_count > 0:
                self._save_candidates(filtered_candidates)
                self.logger.info(f"清理了 {removed_count} 条超过 {days} 天的旧消息")
            
            return removed_count

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
candidate_manager = None  # 候选消息管理器
active_tasks = {}  # 存储活跃任务状态

def init_watermark_tool():
    """初始化水印工具和候选消息管理器"""
    global watermark_tool, candidate_manager
    try:
        logger.info("正在初始化 WatermarkTool...")
        watermark_tool = WatermarkTool()
        logger.info("✅ WatermarkTool 初始化成功")
        
        logger.info("正在初始化候选消息管理器...")
        candidate_manager = CandidateMessageManager()
        logger.info("✅ 候选消息管理器初始化成功")
        
        return True
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
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
        # 🆕 使用标准化的文件名格式，便于后续查找
        name, ext = os.path.splitext(filename)
        # 格式: task_id_modality_upload.ext
        standard_filename = f"{task_id}_{modality}_upload{ext}"
        file_path = UPLOAD_FOLDER / standard_filename
        file.save(str(file_path))
        logger.info(f"上传文件已保存: {file_path}")
        
        # 🆕 对视频文件进行浏览器兼容性转码
        if modality == 'video':
            try:
                logger.info(f"[{task_id}] 开始对上传的视频进行浏览器兼容性转码")
                
                # 检查是否需要转码
                if not VideoTranscoder.is_web_compatible(str(file_path)):
                    # 生成转码后的文件路径
                    transcoded_path = file_path.parent / f"{task_id}_{modality}_upload_web_compatible.mp4"
                    
                    # 执行转码
                    result_path = VideoTranscoder.transcode_for_browser(
                        input_path=str(file_path),
                        output_path=str(transcoded_path),
                        target_fps=15,
                        quality='medium'
                    )
                    
                    logger.info(f"[{task_id}] 视频转码完成: {result_path}")
                    
                    # 删除原始文件，返回转码后的文件
                    try:
                        os.remove(str(file_path))
                        logger.info(f"[{task_id}] 已删除原始上传文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"[{task_id}] 删除原始文件失败: {e}")
                    
                    return result_path
                else:
                    logger.info(f"[{task_id}] 上传的视频已经是浏览器兼容格式，无需转码")
                    
            except Exception as e:
                logger.error(f"[{task_id}] 视频转码失败: {e}")
                logger.info(f"[{task_id}] 使用原始文件")
        
        return str(file_path)
    return ""

def get_file_metadata(file_path: Union[str, Path], modality: str) -> Dict[str, Any]:
    """获取文件元数据信息"""
    try:
        if not file_path or not Path(file_path).exists():
            return {}
        
        file_path = Path(file_path)
        metadata = {
            "filename": file_path.name,
            "filesize": format_file_size(file_path.stat().st_size),
            "format": file_path.suffix.lower()
        }
        
        if modality == 'image':
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    metadata["resolution"] = f"{img.width}x{img.height}"
                    metadata["mode"] = img.mode
            except Exception:
                pass
        elif modality in ['audio', 'video']:
            # 可以添加更多媒体文件元数据获取
            # 例如使用 ffmpeg-python 或 mutagen
            pass
        
        return metadata
    except Exception as e:
        logger.error(f"获取文件元数据失败: {e}")
        return {}

def get_original_file_path(task_id: str, modality: str, result: Any) -> Optional[str]:
    """获取或创建原始文件路径 (AI生成模式)"""
    try:
        # 为AI生成的内容创建原始版本文件
        
        if modality == 'image':
            # 🆕 检查是否有保存的原始图像文件
            original_path = OUTPUT_FOLDER / f"{task_id}_original_image.png"
            if original_path.exists():
                logger.info(f"找到原始图像文件: {original_path}")
                return str(original_path)
            else:
                logger.warning(f"原始图像文件不存在: {original_path}")
                return None
        elif modality == 'audio':
            # 对于音频，检查是否有原始音频文件
            original_path = OUTPUT_FOLDER / f"{task_id}_original_audio.wav"
            if original_path.exists():
                return str(original_path)
            return None  # 暂未实现
        elif modality == 'video':
            # 🆕 对于视频，检查是否有保存的原始视频文件
            original_path = OUTPUT_FOLDER / f"{task_id}_original_video.mp4"
            if original_path.exists():
                logger.info(f"找到原始视频文件: {original_path}")
                return str(original_path)
            else:
                logger.warning(f"原始视频文件不存在: {original_path}")
                return None
        
        return None
    except Exception as e:
        logger.error(f"获取原始文件路径失败: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def send_file_with_range(file_path: Union[str, Path]) -> Response:
    """支持 Range 请求的媒体文件发送，用于音视频流式播放"""
    try:
        path = Path(file_path)
        if not path.exists():
            return jsonify({"error": "文件不存在"}), 404

        file_size = path.stat().st_size
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = 'application/octet-stream'

        range_header = request.headers.get('Range', None)
        if range_header:
            # 解析 Range: bytes=start-end
            try:
                bytes_unit, bytes_range = range_header.split('=')
                start_str, end_str = bytes_range.split('-')
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                # 边界检查
                start = max(0, start)
                end = min(end, file_size - 1)
                if start > end:
                    start = 0
                    end = file_size - 1

                length = end - start + 1
                with open(path, 'rb') as f:
                    f.seek(start)
                    data = f.read(length)
                rv = Response(data, 206, mimetype=mime_type, direct_passthrough=True)
                rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
                rv.headers.add('Accept-Ranges', 'bytes')
                rv.headers.add('Content-Length', str(length))
                return rv
            except Exception:
                # 回退到完整文件
                pass

        # 非 Range 请求，直接返回完整文件
        return send_file(str(path), mimetype=mime_type, as_attachment=False)
    except Exception as e:
        logger.error(f"发送媒体文件失败: {e}")
        return jsonify({"error": f"获取文件失败: {str(e)}"}), 500

def _guess_media_kind_by_suffix(file_path: Union[str, Path]) -> str:
    """根据文件后缀猜测媒体类型: 'video' | 'audio' | 'image' | 'other'"""
    suffix = str(file_path).lower()
    if suffix.endswith(('.mp4', '.webm', '.ogg', '.avi', '.mov', '.mkv', '.flv')):
        return 'video'
    if suffix.endswith(('.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg')):
        return 'audio'
    if suffix.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
        return 'image'
    return 'other'

def _find_original_file_by_task(task_id: str) -> Optional[str]:
    """当内存任务状态缺失时，通过约定文件名查找原始文件"""
    try:
        # 优先查找生成模式的原始文件
        for p in OUTPUT_FOLDER.glob(f"{task_id}_original_*"):
            if p.is_file():
                return str(p)
        # 再查找上传模式的原始上传文件（兼容 *_upload_web_compatible 等）
        for p in UPLOAD_FOLDER.glob(f"{task_id}_*_upload*"):
            if p.is_file():
                return str(p)
    except Exception as e:
        logger.warning(f"通过glob查找原始文件失败: {e}")
    return None

def _find_watermarked_file_by_task(task_id: str) -> Optional[str]:
    """当内存任务状态缺失时，通过约定文件名查找水印文件"""
    try:
        for p in OUTPUT_FOLDER.glob(f"{task_id}_watermarked_*"):
            if p.is_file():
                return str(p)
    except Exception as e:
        logger.warning(f"通过glob查找水印文件失败: {e}")
    return None

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
            
            # 🆕 保存候选消息到候选消息管理器
            try:
                # 获取编码后的binary数据
                text_watermark = watermark_tool.engine._text_watermark
                if text_watermark and hasattr(text_watermark, 'get_last_encoded_binary'):
                    encoded_binary = text_watermark.get_last_encoded_binary()
                    if encoded_binary:
                        message_id = candidate_manager.save_message(
                            original_message=message,
                            encoded_binary=encoded_binary,
                            task_id=task_id,
                            modality=modality
                        )
                        logger.info(f"[{task_id}] 保存候选消息: {message_id}")
                    else:
                        logger.warning(f"[{task_id}] 未获取到编码binary数据")
                else:
                    logger.warning(f"[{task_id}] 文本水印处理器不可用或不支持获取编码数据")
            except Exception as e:
                logger.error(f"[{task_id}] 保存候选消息失败: {e}")
                # 不影响主流程，继续执行
            
            # 保存生成的文本
            output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_text.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(result))
                
        elif modality == 'image':
            # 图像水印嵌入
            active_tasks[task_id]["progress"] = 50
            
            # 检查是否为上传模式
            upload_mode = request.form.get('upload_mode', 'false').lower() == 'true'
            
            if upload_mode:
                # 上传文件模式：为现有图像添加水印
                file = request.files.get('file')
                if not file or not file.filename:
                    raise ValueError("上传模式需要提供图像文件")
                
                # 验证文件类型
                if not allowed_file(file.filename, modality):
                    raise ValueError(f"不支持的{modality}文件类型: {file.filename}")
                
                # 保存上传的文件
                upload_file_path = save_uploaded_file(file, task_id, modality)
                logger.info(f"[{task_id}] 上传图像文件: {upload_file_path}")
                
                # 使用 image_input 参数传递现有图像
                result = watermark_tool.embed("uploaded image", message, 'image', 
                                            image_input=upload_file_path)
            else:
                # 生成模式：AI生成图像并嵌入水印
                if not prompt.strip():
                    raise ValueError("生成模式需要提供图像描述提示词")
                
                # 获取额外参数
                resolution = int(request.form.get('resolution', 512))
                num_inference_steps = int(request.form.get('num_inference_steps', 30))
                
                result = watermark_tool.embed(prompt, message, 'image', 
                                            resolution=resolution,
                                            num_inference_steps=num_inference_steps)
            
            # 🆕 处理新的返回格式
            if isinstance(result, dict) and 'watermarked' in result:
                # AI生成模式：result包含original和watermarked图像
                watermarked_image = result['watermarked']
                original_image = result['original']
                
                # 保存水印图像
                output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_image.png"
                if hasattr(watermarked_image, 'save'):
                    watermarked_image.save(str(output_path))
                
                # 🆕 保存原始图像
                original_path = OUTPUT_FOLDER / f"{task_id}_original_image.png"
                if hasattr(original_image, 'save'):
                    original_image.save(str(original_path))
                    logger.info(f"[{task_id}] 原始图像已保存: {original_path}")
            else:
                # 上传模式或旧格式：直接保存result
                output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_image.png"
                if hasattr(result, 'save'):  # PIL Image
                    result.save(str(output_path))
            
        elif modality == 'audio':
            # 音频水印嵌入
            active_tasks[task_id]["progress"] = 50
            
            # 检查是否为上传模式
            upload_mode = request.form.get('upload_mode', 'false').lower() == 'true'
            
            output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_audio.wav"
            
            if upload_mode:
                # 上传文件模式：为现有音频添加水印
                file = request.files.get('file')
                if not file or not file.filename:
                    raise ValueError("上传模式需要提供音频文件")
                
                # 验证文件类型
                if not allowed_file(file.filename, modality):
                    raise ValueError(f"不支持的{modality}文件类型: {file.filename}")
                
                # 保存上传的文件
                upload_file_path = save_uploaded_file(file, task_id, modality)
                logger.info(f"[{task_id}] 上传音频文件: {upload_file_path}")
                
                # 使用 audio_input 参数传递现有音频
                result = watermark_tool.embed("uploaded audio", message, 'audio',
                                            audio_input=upload_file_path,
                                            output_path=str(output_path))
            else:
                # 生成模式：文本转语音并嵌入水印
                if not prompt.strip():
                    raise ValueError("生成模式需要提供音频内容提示词")
                
                # 获取额外参数
                voice_preset = request.form.get('voice_preset', 'v2/en_speaker_6')
                alpha = float(request.form.get('alpha', 1.0))
                
                result = watermark_tool.embed(prompt, message, 'audio',
                                            voice_preset=voice_preset,
                                            alpha=alpha,
                                            output_path=str(output_path))
                
                # 🆕 处理新的返回格式（AI生成音频）
                if not upload_mode:
                    if isinstance(result, dict) and 'watermarked' in result:
                        watermarked_audio_path = result.get('watermarked')
                        original_audio_path = result.get('original')

                        # 将水印后音频移动到标准输出路径
                        if watermarked_audio_path and watermarked_audio_path != str(output_path) and os.path.exists(watermarked_audio_path):
                            shutil.move(watermarked_audio_path, str(output_path))
                            logger.info(f"[{task_id}] 水印音频已移动到: {output_path}")

                        # 保存原始音频，供前端对比展示
                        if original_audio_path and os.path.exists(original_audio_path):
                            original_output_path = OUTPUT_FOLDER / f"{task_id}_original_audio.wav"
                            shutil.move(original_audio_path, str(original_output_path))
                            logger.info(f"[{task_id}] 原始音频已保存: {original_output_path}")
            
        elif modality == 'video':
            # 视频水印嵌入
            active_tasks[task_id]["progress"] = 50
            
            # 检查是否为上传模式
            upload_mode = request.form.get('upload_mode', 'false').lower() == 'true'
            
            # 指定输出路径到demo_outputs目录
            output_path = OUTPUT_FOLDER / f"{task_id}_watermarked_video.mp4"
            
            if upload_mode:
                # 上传文件模式：为现有视频添加水印
                file = request.files.get('file')
                if not file or not file.filename:
                    raise ValueError("上传模式需要提供视频文件")
                
                # 验证文件类型
                if not allowed_file(file.filename, modality):
                    raise ValueError(f"不支持的{modality}文件类型: {file.filename}")
                
                # 保存上传的文件
                upload_file_path = save_uploaded_file(file, task_id, modality)
                logger.info(f"[{task_id}] 上传视频文件: {upload_file_path}")
                
                # 使用 video_input 参数传递现有视频
                result = watermark_tool.embed("uploaded video", message, 'video',
                                            video_input=upload_file_path,
                                            output_path=str(output_path))
            else:
                # 生成模式：文生视频并嵌入水印
                if not prompt.strip():
                    raise ValueError("生成模式需要提供视频描述提示词")
                
                # 获取额外参数
                num_frames = int(request.form.get('num_frames', 25))
                resolution = request.form.get('resolution', '512x320')
                width, height = map(int, resolution.split('x'))
                
                result = watermark_tool.embed(prompt, message, 'video',
                                            num_frames=num_frames,
                                            width=width,
                                            height=height,
                                            output_path=str(output_path))
            
            # 🆕 处理新的返回格式
            if isinstance(result, dict) and 'watermarked' in result:
                # AI生成模式：result包含original和watermarked视频路径
                watermarked_video_path = result['watermarked']
                original_video_path = result['original']
                
                # 移动水印视频到指定输出路径
                if watermarked_video_path != str(output_path):
                    if os.path.exists(watermarked_video_path):
                        shutil.move(watermarked_video_path, str(output_path))
                        logger.info(f"水印视频已移动到: {output_path}")
                
                # 🆕 移动原始视频到demo_outputs目录
                if original_video_path and os.path.exists(original_video_path):
                    original_output_path = OUTPUT_FOLDER / f"{task_id}_original_video.mp4"
                    shutil.move(original_video_path, str(original_output_path))
                    logger.info(f"[{task_id}] 原始视频已保存: {original_output_path}")
            else:
                # 上传模式或旧格式：直接移动result文件
                if result != str(output_path) and os.path.exists(result):
                    shutil.move(result, str(output_path))
                    logger.info(f"视频文件已移动到: {output_path}")
            
        else:
            raise ValueError(f"不支持的模态类型: {modality}")
        
        # 获取文件信息和元数据
        file_info = get_file_metadata(output_path, modality) if output_path else {}
        
        # 🆕 增强的任务状态数据
        task_data = {
            "status": "completed",
            "progress": 100,
            "modality": modality,
            "input_mode": "upload" if request.form.get('upload_mode', 'false').lower() == 'true' else "generate",
            "message": message,
            "prompt": request.form.get('prompt', ''),
            "watermarked_file": str(output_path) if output_path else None,
            "original_file": None,  # 将在下面根据模态设置
            "metadata": file_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # 🆕 对于文本模态，添加生成的文本内容到任务数据
        if modality == 'text' and result:
            task_data["generated_text"] = str(result)
            task_data["watermarked_text"] = str(result)  # 文本水印中生成的就是带水印的文本
        
        # 🆕 处理原始文件路径 (AI生成模式需要保存原始版本)
        original_file_path = None
        if task_data["input_mode"] == "generate" and modality in ['image', 'audio', 'video']:
            # AI生成模式：尝试获取原始生成的文件 (未加水印版本)
            original_file_path = get_original_file_path(task_id, modality, result)
            task_data["original_file"] = original_file_path
        elif task_data["input_mode"] == "upload":
            # 上传模式：原始文件就是用户上传(或为浏览器转码后的)文件
            try:
                prefix = f"{task_id}_{modality}_upload"
                # 兼容 *_upload_web_compatible.mp4 等情况
                matches = sorted(UPLOAD_FOLDER.glob(f"{prefix}*"))
                if matches:
                    task_data["original_file"] = str(matches[0])
                    logger.info(f"找到原始上传文件: {task_data['original_file']}")
                else:
                    logger.warning(f"未找到原始上传文件: {UPLOAD_FOLDER}/{prefix}*")
            except Exception as e:
                logger.warning(f"查找原始上传文件出错: {e}")
        
        # 更新任务状态
        active_tasks[task_id] = task_data
        
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
            
            # 🆕 使用候选消息列表进行智能提取
            try:
                # 获取候选消息列表
                candidates = candidate_manager.get_candidates(modality)
                candidate_messages = [item['original_message'] for item in candidates.values()]
                
                logger.info(f"[{task_id}] 使用 {len(candidate_messages)} 个候选消息进行提取")
                
                # 使用候选消息列表进行提取
                if candidate_messages:
                    result = watermark_tool.extract(content, 'text', candidates_messages=candidate_messages)
                else:
                    # 没有候选消息时使用标准提取
                    result = watermark_tool.extract(content, 'text')
                
                # 🆕 如果标准提取失败且有候选消息，尝试直接二进制匹配
                if (not result.get('detected', False) or not result.get('success', False)) and candidates:
                    logger.info(f"[{task_id}] 标准提取失败，尝试候选消息匹配")
                    
                    try:
                        # 获取原始解码的二进制数据
                        raw_binary = result.get('binary_message', [])
                        if raw_binary:
                            best_match, match_score = candidate_manager.find_best_match(
                                decoded_binary=raw_binary,
                                threshold=0.4,  # 40% 匹配阈值
                                modality=modality
                            )
                            
                            if best_match and match_score >= 0.4:
                                # 找到满足阈值的匹配
                                result = {
                                    'detected': True,
                                    'success': True,
                                    'message': best_match,
                                    'confidence': match_score,
                                    'method': 'candidate_matching',
                                    'metadata': {
                                        'algorithm': 'CredID',
                                        'matching_method': 'candidate_matching',
                                        'match_score': match_score,
                                        'original_detected': result.get('detected', False),
                                        'candidates_used': len(candidate_messages)
                                    }
                                }
                                logger.info(f"[{task_id}] 候选匹配成功: '{best_match}' (score: {match_score:.3f})")
                            else:
                                logger.info(f"[{task_id}] 候选匹配未找到满足阈值的结果")
                        else:
                            logger.warning(f"[{task_id}] 未获取到原始二进制数据，无法进行候选匹配")
                    except Exception as match_error:
                        logger.error(f"[{task_id}] 候选匹配过程出错: {match_error}")
                        # 继续使用原始结果
                        
            except Exception as e:
                logger.error(f"[{task_id}] 获取候选消息失败: {e}")
                # 回退到标准提取
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
        
        # 规范化置信度，避免出现超过100%的显示
        confidence_value = result.get('confidence', 0.0)
        try:
            confidence_value = float(confidence_value)
        except (TypeError, ValueError):
            confidence_value = 0.0
        # 如果算法返回的是百分比或异常值，限制在[0, 1]
        if confidence_value > 1.0:
            confidence_value = 1.0
        if confidence_value < 0.0:
            confidence_value = 0.0

        # 构建响应 - 确保所有值都是JSON可序列化的
        response_data = {
            "task_id": task_id,
            "status": "success",
            "modality": modality,
            "detected": bool(result.get('detected', False)),  # 确保是Python原生bool
            "message": str(result.get('message', '')),
            "confidence": confidence_value,
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

@app.route('/api/visible_mark', methods=['POST'])
def api_visible_mark():
    """可见标识添加接口"""
    task_id = generate_task_id()
    logger.info(f"[{task_id}] 🏷️ 开始处理可见标识请求")
    
    # 初始化任务状态
    active_tasks[task_id] = {
        "status": "processing",
        "timestamp": datetime.now().isoformat(),
        "progress": 0,
        "modality": None,
        "operation": "visible_mark"
    }
    
    try:
        # 获取参数
        modality = request.form.get('modality')
        mark_text = request.form.get('mark_text', '本内容由人工智能生成/合成')
        
        if not modality or modality not in ['text', 'image', 'audio', 'video']:
            return jsonify({"error": "无效的模态类型"}), 400
        
        active_tasks[task_id]["modality"] = modality
        active_tasks[task_id]["progress"] = 10
        
        logger.info(f"[{task_id}] 📝 模态: {modality}, 标识文本: {mark_text}")
        
        # 导入可见标识模块
        from src.utils.visible_mark import (
            add_text_mark_to_text,
            add_overlay_to_image, 
            add_overlay_to_video_ffmpeg,
            add_voice_mark_to_audio
        )
        from PIL import Image
        
        if modality == 'text':
            # 文本标识处理
            text_content = request.form.get('text', '')
            position = request.form.get('position', 'start')
            
            if not text_content.strip():
                return jsonify({"error": "文本内容不能为空"}), 400
            
            logger.info(f"[{task_id}] 文本标识参数: 位置={position}, 文案={mark_text}, 原文长度={len(text_content)}")
            
            active_tasks[task_id]["progress"] = 50
            
            # 添加文本标识
            marked_text = add_text_mark_to_text(text_content, mark_text, position)
            
            active_tasks[task_id]["progress"] = 90
            
            # 保存结果到文件
            output_dir = Path("demo_outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"task_{task_id}_marked_text.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(marked_text)
            
            # 更新任务状态
            active_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "output_path": str(output_path),
                "output_text": marked_text,
                "watermarked_file": str(output_path)
            })
            
            logger.info(f"[{task_id}] ✅ 文本标识添加完成")
            
            return jsonify({
                "task_id": task_id,
                "status": "completed",
                "output_text": marked_text,
                "output_path": str(output_path),
                "timestamp": datetime.now().isoformat()
            })
            
        else:
            # 文件上传处理
            uploaded_file = request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '':
                return jsonify({"error": "未上传文件"}), 400
            
            # 保存上传文件
            uploads_dir = Path("demo_uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            filename = secure_filename(uploaded_file.filename)
            input_path = uploads_dir / f"{task_id}_{filename}"
            uploaded_file.save(input_path)
            
            active_tasks[task_id]["progress"] = 30
            
            # 输出目录
            output_dir = Path("demo_outputs")
            output_dir.mkdir(exist_ok=True)
            
            if modality == 'image':
                # 图像标识处理
                position = request.form.get('position', 'bottom_right')
                font_percent = float(request.form.get('font_percent', 5.0))
                font_color = request.form.get('font_color', '#FFFFFF')
                
                logger.info(f"[{task_id}] 图像标识参数: 位置={position}, 字号={font_percent}%, 颜色={font_color}, 文案={mark_text}")
                
                # 打开图像
                img = Image.open(input_path)
                active_tasks[task_id]["progress"] = 60
                
                # 添加图像标识（无背景框版本）
                marked_img = add_overlay_to_image(
                    img, mark_text, position, font_percent, font_color, bg_rgba=None
                )
                active_tasks[task_id]["progress"] = 80
                
                # 保存结果
                output_path = output_dir / f"task_{task_id}_marked_image.{img.format.lower() if img.format else 'png'}"
                marked_img.save(output_path)
                
                active_tasks[task_id].update({
                    "status": "completed",
                    "progress": 100,
                    "output_path": str(output_path),
                    "watermarked_file": str(output_path),
                    "original_file": str(input_path)
                })
                
            elif modality == 'video':
                # 视频标识处理
                position = request.form.get('position', 'bottom_right')
                font_percent = float(request.form.get('font_percent', 5.0))
                duration_seconds = float(request.form.get('duration_seconds', 2.0))
                font_color = request.form.get('font_color', 'white')
                
                logger.info(f"[{task_id}] 视频标识参数: 位置={position}, 字号={font_percent}%, 颜色={font_color}, 时长={duration_seconds}s, 文案={mark_text}")
                
                active_tasks[task_id]["progress"] = 60
                
                # 输出文件路径
                output_path = output_dir / f"task_{task_id}_marked_video.mp4"
                
                # 添加视频标识（无背景框版本）
                result_path = add_overlay_to_video_ffmpeg(
                    str(input_path), str(output_path), mark_text, position,
                    font_percent, duration_seconds, font_color, box_color="transparent"
                )
                
                active_tasks[task_id]["progress"] = 90
                
                # 浏览器兼容性检查 + 转码（避免调用不存在的方法）
                final_output = str(result_path)
                try:
                    if not VideoTranscoder.is_web_compatible(final_output):
                        web_path = output_dir / f"task_{task_id}_marked_video_web.mp4"
                        final_output = VideoTranscoder.transcode_for_browser(
                            input_path=str(result_path),
                            output_path=str(web_path),
                            target_fps=15,
                            quality='medium'
                        )
                except Exception as e:
                    logger.warning(f"[{task_id}] 浏览器兼容性处理失败，使用原视频: {e}")
                
                active_tasks[task_id].update({
                    "status": "completed", 
                    "progress": 100,
                    "output_path": final_output,
                    "watermarked_file": final_output,
                    "original_file": str(input_path)
                })
                
            elif modality == 'audio':
                # 音频标识处理
                position = request.form.get('position', 'start')
                voice_preset = request.form.get('voice_preset', 'v2/zh_speaker_6')
                
                logger.info(f"[{task_id}] 音频标识参数: 位置={position}, 语音预设={voice_preset}, 文案={mark_text}")
                
                active_tasks[task_id]["progress"] = 60
                
                # 输出文件路径
                output_path = output_dir / f"task_{task_id}_marked_audio.wav"
                
                # 添加音频标识
                result_path = add_voice_mark_to_audio(
                    str(input_path), str(output_path), mark_text, position, voice_preset
                )
                
                active_tasks[task_id].update({
                    "status": "completed",
                    "progress": 100,
                    "output_path": result_path,
                    "watermarked_file": result_path,
                    "original_file": str(input_path)
                })
            
            logger.info(f"[{task_id}] ✅ {modality} 标识添加完成: {active_tasks[task_id]['output_path']}")
            
            return jsonify({
                "task_id": task_id,
                "status": "completed",
                "output_path": active_tasks[task_id]["output_path"],
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        active_tasks[task_id]["status"] = "error"
        active_tasks[task_id]["error"] = str(e)
        
        logger.error(f"[{task_id}] ❌ 可见标识添加失败: {e}")
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
        # 查找输出文件 - 支持多种命名模式
        output_files = []
        
        # 模式1: 标准水印文件 {task_id}_*
        output_files.extend(list(OUTPUT_FOLDER.glob(f"{task_id}_*")))
        
        # 模式2: 可见标识文件 task_{task_id}_marked_*
        output_files.extend(list(OUTPUT_FOLDER.glob(f"task_{task_id}_marked_*")))
        
        if not output_files:
            logger.error(f"未找到任务 {task_id} 的结果文件")
            return jsonify({"error": "未找到结果文件"}), 404
        
        # 返回第一个匹配的文件，优先返回水印后的文件
        # 对于可见标识，查找 marked_ 文件
        watermarked_files = [f for f in output_files if 'watermarked' in f.name or 'marked' in f.name]
        if watermarked_files:
            file_path = watermarked_files[0]
        else:
            file_path = output_files[0]
        
        logger.info(f"下载文件: {file_path}")
        return send_file(str(file_path), as_attachment=True)
        
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        return jsonify({"error": f"下载失败: {str(e)}"}), 500

@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    """获取任务状态 - 增强版本，包含文件URL和元数据"""
    if task_id not in active_tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task_data = active_tasks[task_id].copy()
    
    # 🆕 添加文件访问URL
    if task_data.get("status") == "completed":
        files_info = {}
        
        if task_data.get("original_file"):
            files_info["original"] = f"/api/files/{task_id}/original"
        if task_data.get("watermarked_file"):
            files_info["watermarked"] = f"/api/files/{task_id}/watermarked"
        
        task_data["files"] = files_info
    
    return jsonify(task_data)

@app.route('/api/files/<task_id>/original')
def get_original_file(task_id):
    """获取原始文件 - 用于对比展示"""
    try:
        if task_id not in active_tasks:
            return jsonify({"error": "任务不存在"}), 404
        
        task_data = active_tasks.get(task_id, {})
        original_file = task_data.get("original_file")
        if not original_file:
            # 回退：通过文件名约定查找
            original_file = _find_original_file_by_task(task_id)
        
        if not original_file or not Path(original_file).exists():
            return jsonify({"error": "原始文件不存在"}), 404
        
        # 对音视频使用 Range 支持
        kind = _guess_media_kind_by_suffix(original_file)
        if kind in ('video', 'audio'):
            return send_file_with_range(original_file)
        return send_file(original_file, as_attachment=False)
    except Exception as e:
        logger.error(f"获取原始文件失败: {e}")
        return jsonify({"error": f"获取文件失败: {str(e)}"}), 500

@app.route('/api/files/<task_id>/watermarked')
def get_watermarked_file(task_id):
    """获取水印文件 - 用于对比展示"""
    try:
        if task_id not in active_tasks:
            return jsonify({"error": "任务不存在"}), 404
        
        task_data = active_tasks.get(task_id, {})
        watermarked_file = task_data.get("watermarked_file")
        if not watermarked_file:
            # 回退：通过文件名约定查找
            watermarked_file = _find_watermarked_file_by_task(task_id)
        
        if not watermarked_file or not Path(watermarked_file).exists():
            return jsonify({"error": "水印文件不存在"}), 404
        
        # 对音视频使用 Range 支持
        kind = _guess_media_kind_by_suffix(watermarked_file)
        if kind in ('video', 'audio'):
            return send_file_with_range(watermarked_file)
        return send_file(watermarked_file, as_attachment=False)
    except Exception as e:
        logger.error(f"获取水印文件失败: {e}")
        return jsonify({"error": f"获取文件失败: {str(e)}"}), 500

@app.route('/api/candidates', methods=['GET'])
def api_candidates():
    """获取候选消息列表和统计信息"""
    try:
        if not candidate_manager:
            raise Exception("候选消息管理器未初始化")
        
        # 获取模态类型参数
        modality = request.args.get('modality', 'text')
        
        # 获取候选消息和统计信息
        candidates = candidate_manager.get_candidates(modality)
        stats = candidate_manager.get_statistics()
        
        # 构建响应
        response_data = {
            "status": "success",
            "modality": modality,
            "candidates": candidates,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"获取候选消息失败: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/candidates/clear', methods=['POST'])
def api_clear_candidates():
    """清理旧的候选消息"""
    try:
        if not candidate_manager:
            raise Exception("候选消息管理器未初始化")
        
        # 获取清理天数参数
        days = int(request.form.get('days', 30))
        
        # 清理旧消息
        removed_count = candidate_manager.clear_old_messages(days)
        
        return jsonify({
            "status": "success",
            "removed_count": removed_count,
            "days": days,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"清理候选消息失败: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

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
        host='0.0.0.0',  # 监听所有网络接口
        port=5000,
        debug=True,
        threaded=True
    )