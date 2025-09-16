"""
多模态水印工具统一引擎
遵循KISS原则，提供简洁统一的多模态水印接口
"""

import torch
import logging
from typing import Dict, Any, Optional, Union
from PIL import Image

try:
    # 相对导入（当作为包运行时）
    from ..text_watermark.credid_watermark import CredIDWatermark
    from ..image_watermark.image_watermark import ImageWatermark
    from ..audio_watermark.audio_watermark import AudioWatermark
    from ..video_watermark.video_watermark import VideoWatermark
except ImportError:
    try:
        # 绝对导入（当 src 在路径中时）
        from text_watermark.credid_watermark import CredIDWatermark
        from image_watermark.image_watermark import ImageWatermark
        from audio_watermark.audio_watermark import AudioWatermark
        from video_watermark.video_watermark import VideoWatermark
    except ImportError as e:
        raise ImportError(f"无法导入水印模块: {e}. 请确保从项目根目录运行，并且 src 目录在 Python 路径中。")


class UnifiedWatermarkEngine:
    """
    多模态水印统一引擎
    
    遵循KISS原则的简洁设计：
    - 统一的embed/extract接口
    - 使用测试验证的最优默认参数
    - 图像默认使用videoseal算法
    - 支持text/image/audio/video四种模态
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化统一水印引擎
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        
        # 延迟初始化各模态处理器，节省内存
        self._text_watermark = None
        self._image_watermark = None  
        self._audio_watermark = None
        self._video_watermark = None
        # 文本模型与分词器（懒加载后缓存）
        self._text_model = None
        self._text_tokenizer = None
        
        self.config_path = config_path
        
        self.logger.info("UnifiedWatermarkEngine初始化完成")
    
    def _project_root(self) -> str:
        """获取项目根目录（基于当前文件位置推断）。"""
        import os
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def _candidate_cache_dirs(self) -> list:
        """返回可能的本地缓存目录列表（按优先级）。"""
        import os
        candidates = []
        if os.getenv('HF_HOME'):
            candidates.append(os.path.join(os.getenv('HF_HOME'), 'hub'))
        if os.getenv('HF_HUB_CACHE'):
            candidates.append(os.getenv('HF_HUB_CACHE'))
        # 项目内 models 目录
        candidates.append(os.path.join(self._project_root(), 'models'))
        # 用户级默认缓存
        candidates.append('/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub')
        candidates.append(os.path.expanduser('~/.cache/huggingface/hub'))
        # 去重并保留顺序
        seen = set()
        ordered = []
        for p in candidates:
            if p and p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    def _load_text_config(self) -> Dict[str, Any]:
        """加载文本水印配置。优先使用传入的 config_path，其次使用项目默认。"""
        import os
        import yaml
        # 优先使用 self.config_path
        cfg_path = None
        if self.config_path and os.path.isfile(self.config_path):
            cfg_path = self.config_path
        else:
            # 默认指向项目内 config/text_config.yaml
            default_path = os.path.join(self._project_root(), 'config', 'text_config.yaml')
            if os.path.isfile(default_path):
                cfg_path = default_path
        if cfg_path is None:
            # 退回到内置默认
            return {
                'mode': 'lm',
                'model_name': 'sshleifer/tiny-gpt2',
                'lm_params': {
                    'delta': 1.5,
                    'prefix_len': 10,
                    'message_len': 10
                },
                'wm_params': {
                    'encode_ratio': 8,
                    'strategy': 'vanilla'
                },
                'model_config': {
                    'cache_dir': os.path.join(self._project_root(), 'models')
                }
            }
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data

    def _init_text_model_tokenizer(self):
        """使用与 test_complex_messages_real.py 一致的策略初始化文本模型与分词器（离线优先）。"""
        if self._text_model is not None and self._text_tokenizer is not None:
            return
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 强制离线首选，避免联网依赖
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        cfg = self._load_text_config()
        primary_model = cfg.get('model_name', 'sshleifer/tiny-gpt2')
        model_cfg = cfg.get('model_config', {})

        # 构造候选模型列表：优先配置，其次tiny模型
        candidate_models = [m for m in [primary_model, 'sshleifer/tiny-gpt2'] if m]

        # 遍历可能的缓存目录并尝试加载
        candidate_cache_dirs = []
        if model_cfg.get('cache_dir'):
            candidate_cache_dirs.append(model_cfg.get('cache_dir'))
        candidate_cache_dirs.extend(self._candidate_cache_dirs())

        trust_remote_code = bool(model_cfg.get('trust_remote_code', True))
        last_error = None

        for model_name in candidate_models:
            for cache_dir in candidate_cache_dirs:
                try:
                    if cache_dir and not os.path.isdir(cache_dir):
                        continue
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=trust_remote_code,
                        use_fast=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=trust_remote_code,
                        device_map=model_cfg.get('device_map', 'auto'),
                        torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    self._text_model = model
                    self._text_tokenizer = tokenizer
                    self.logger.info(f"文本模型加载成功: {model_name} (cache_dir={cache_dir})")
                    return
                except Exception as e:
                    last_error = e
                    continue

        # 若全部失败，记录警告
        self.logger.warning(f"离线加载文本模型失败，稍后在调用时仍将报错。最后错误: {last_error}")

    def _get_text_watermark(self) -> CredIDWatermark:
        """获取文本水印处理器（懒加载）"""
        if self._text_watermark is None:
            # 读取配置并初始化处理器
            config = self._load_text_config()
            self._text_watermark = CredIDWatermark(config)
            # 同步初始化模型与分词器（离线优先）
            self._init_text_model_tokenizer()
        return self._text_watermark
    
    def _get_image_watermark(self) -> ImageWatermark:
        """获取图像水印处理器（懒加载）"""
        if self._image_watermark is None:
            self._image_watermark = ImageWatermark(self.config_path)
            # 设置为videoseal算法（默认）
            self._image_watermark.algorithm = 'videoseal'
        return self._image_watermark
    
    def _get_audio_watermark(self) -> AudioWatermark:
        """获取音频水印处理器（懒加载）"""
        if self._audio_watermark is None:
            self._audio_watermark = AudioWatermark(self.config_path)
        return self._audio_watermark
    
    def _get_video_watermark(self) -> VideoWatermark:
        """获取视频水印处理器（懒加载）"""
        if self._video_watermark is None:
            from ..video_watermark.video_watermark import create_video_watermark
            self._video_watermark = create_video_watermark()
        return self._video_watermark
    
    def embed(self, prompt: str, message: str, modality: str, **kwargs) -> Any:
        """
        统一嵌入接口
        
        Args:
            prompt: 输入提示（文本内容、图像描述、音频文本等）
            message: 要嵌入的水印消息
            modality: 模态类型 ('text', 'image', 'audio', 'video')
            **kwargs: 额外参数（如model, tokenizer等）
            
        Returns:
            带水印的内容（具体类型取决于模态）
            - text: str
            - image: PIL.Image
            - audio: torch.Tensor 或 str（如果指定output_path）
            - video: str（视频文件路径）
        """
        self.logger.info(f"开始{modality}水印嵌入: prompt='{prompt[:50]}...', message='{message}'")
        
        try:
            if modality == 'text':
                # 文本水印：需要模型和分词器
                watermark = self._get_text_watermark()
                
                # CredID需要模型和分词器参数
                model = kwargs.get('model') or self._text_model
                tokenizer = kwargs.get('tokenizer') or self._text_tokenizer
                
                if model is None or tokenizer is None:
                    raise ValueError("文本水印需要提供model和tokenizer参数")
                
                # 调用正确的embed方法
                result = watermark.embed(model, tokenizer, prompt, message)
                
                if result.get('success'):
                    return result['watermarked_text']
                else:
                    raise RuntimeError(f"文本水印嵌入失败: {result.get('error', 'Unknown error')}")
                    
                
            elif modality == 'image':
                # 图像水印：使用videoseal算法
                watermark = self._get_image_watermark()
                if 'image_input' in kwargs:
                    # 在现有图像上嵌入水印
                    image_input = kwargs.pop('image_input')  # 移除避免重复传递
                    return watermark.embed_watermark(
                        image_input, 
                        message=message, 
                        **kwargs
                    )
                else:
                    # 生成新图像并嵌入水印
                    # 🆕 AI生成模式：请求返回原始图像
                    return watermark.generate_with_watermark(
                        prompt, 
                        message=message,
                        return_original=True,  # 请求同时返回原始图像
                        **kwargs
                    )
                    
            elif modality == 'audio':
                # 音频水印：使用audioseal算法
                watermark = self._get_audio_watermark()
                if 'audio_input' in kwargs:
                    # 在现有音频上嵌入水印
                    audio_input = kwargs.pop('audio_input')  # 移除audio_input避免重复
                    return watermark.embed_watermark(
                        audio_input, 
                        message, 
                        **kwargs
                    )
                else:
                    # 文本转语音+水印
                    # 🆕 对于AI生成音频，传递 return_original=True 以支持对比显示
                    return watermark.generate_audio_with_watermark(
                        prompt, 
                        message,
                        return_original=True,
                        **kwargs
                    )
                    
            elif modality == 'video':
                # 视频水印：HunyuanVideo + VideoSeal
                watermark = self._get_video_watermark()
                if 'video_input' in kwargs:
                    # 在现有视频上嵌入水印
                    video_input = kwargs.pop('video_input')  # 移除video_input避免重复
                    return watermark.embed_watermark(
                        video_input, 
                        message, 
                        **kwargs
                    )
                else:
                    # 文生视频+水印
                    # 若未传入分辨率，设置更安全的默认分辨率（16倍数）
                    if 'height' not in kwargs:
                        kwargs['height'] = 320
                    if 'width' not in kwargs:
                        kwargs['width'] = 512
                    # 🆕 AI生成模式：请求返回原始视频
                    return watermark.generate_video_with_watermark(
                        prompt, 
                        message,
                        return_original=True,  # 请求同时返回原始视频
                        **kwargs
                    )
            else:
                raise ValueError(f"不支持的模态类型: {modality}")
                
        except Exception as e:
            self.logger.error(f"{modality}水印嵌入失败: {e}")
            raise
    
    def extract(self, content: Any, modality: str, **kwargs) -> Dict[str, Any]:
        """
        统一提取接口
        
        Args:
            content: 待检测内容
                - text: str
                - image: PIL.Image 或 str（文件路径）
                - audio: torch.Tensor 或 str（文件路径）
                - video: str（文件路径）
            modality: 模态类型 ('text', 'image', 'audio', 'video')
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 统一格式的检测结果
                - detected: bool, 是否检测到水印
                - message: str, 提取的消息
                - confidence: float, 置信度 (0.0-1.0)
                - metadata: dict, 额外信息
        """
        self.logger.info(f"开始{modality}水印提取")
        
        try:
            if modality == 'text':
                watermark = self._get_text_watermark()
                
                # CredID需要模型和分词器参数
                model = kwargs.get('model') or self._text_model
                tokenizer = kwargs.get('tokenizer') or self._text_tokenizer
                
                if model is None or tokenizer is None:
                    raise ValueError("文本水印提取需要提供model和tokenizer参数")
                
                # 调用正确的extract方法
                result = watermark.extract(content, model, tokenizer, 
                                         candidates_messages=kwargs.get('candidates_messages'))
                
                # 统一返回格式
                return {
                    'detected': result.get('success', False),
                    'message': result.get('extracted_message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }
                
            elif modality == 'image':
                watermark = self._get_image_watermark()
                # 使用优化的VideoSeal参数：replicate=32提高多帧平均稳定性，chunk_size=16优化分块处理
                result = watermark.extract_watermark(
                    content, 
                    replicate=kwargs.get('replicate', 32),
                    chunk_size=kwargs.get('chunk_size', 16),
                    **kwargs
                )
                return {
                    'detected': result.get('detected', False),
                    'message': result.get('message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }
                
            elif modality == 'audio':
                watermark = self._get_audio_watermark()
                result = watermark.extract_watermark(content, **kwargs)
                return {
                    'detected': result.get('detected', False),
                    'message': result.get('message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }
                
            elif modality == 'video':
                watermark = self._get_video_watermark()
                # 使用测试验证的默认参数
                result = watermark.extract_watermark(
                    content,
                    chunk_size=kwargs.get('chunk_size', 16),
                    **kwargs
                )
                return {
                    'detected': result.get('detected', False),
                    'message': result.get('message', ''),
                    'confidence': result.get('confidence', 0.0),
                    'metadata': result.get('metadata', {})
                }
            else:
                raise ValueError(f"不支持的模态类型: {modality}")
                
        except Exception as e:
            self.logger.error(f"{modality}水印提取失败: {e}")
            return {
                'detected': False,
                'message': '',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def get_supported_modalities(self) -> list:
        """获取支持的模态列表"""
        return ['text', 'image', 'audio', 'video']
    
    def get_default_algorithms(self) -> Dict[str, str]:
        """获取各模态的默认算法"""
        return {
            'text': 'credid',
            'image': 'videoseal',  # 默认使用videoseal
            'audio': 'audioseal',
            'video': 'hunyuan+videoseal'
        }


# 便捷工厂函数
def create_unified_engine(config_path: Optional[str] = None) -> UnifiedWatermarkEngine:
    """
    创建统一水印引擎的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        UnifiedWatermarkEngine: 统一水印引擎实例
    """
    return UnifiedWatermarkEngine(config_path)


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    engine = create_unified_engine()
    
    print("支持的模态:", engine.get_supported_modalities())
    print("默认算法:", engine.get_default_algorithms())
    
    print("UnifiedWatermarkEngine测试完成")