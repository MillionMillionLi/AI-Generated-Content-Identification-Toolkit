"""
视频水印模块统一入口
支持HunyuanVideo文生视频 + VideoSeal水印技术
"""

from .video_watermark import VideoWatermark
from .model_manager import ModelManager
from .hunyuan_video_generator import HunyuanVideoGenerator
from .videoseal_wrapper import VideoSealWrapper

__version__ = "1.0.0"
__all__ = ["VideoWatermark", "ModelManager", "HunyuanVideoGenerator", "VideoSealWrapper"]

# 为方便使用，提供快速加载接口
def load_video_watermark(cache_dir="/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"):
    """
    快速加载视频水印工具
    
    Args:
        cache_dir: HuggingFace模型缓存目录
        
    Returns:
        VideoWatermark: 视频水印工具实例
    """
    return VideoWatermark(cache_dir=cache_dir)