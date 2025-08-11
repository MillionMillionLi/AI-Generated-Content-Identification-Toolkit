"""
图像水印类 - 负责图像水印的嵌入和提取
"""

import torch
import yaml
from PIL import Image
from typing import Dict, Any, Optional, Union
import os
from .prc_watermark import PRCWatermark


class ImageWatermark:
    """图像水印处理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化图像水印处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.algorithm = self.config.get('algorithm', 'prc')
        self._setup_model()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 使用默认配置
            return {
                'algorithm': 'prc',
                'watermark_key': 'default',
                'model_name': 'stabilityai/stable-diffusion-2-1-base',
                'false_positive_rate': 1e-5,
                'prc_t': 3,
                'n': 4 * 64 * 64,
                'keys_dir': 'keys'
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('image_watermark', {})
    
    def _setup_model(self):
        """初始化模型"""
        print(f"Setting up {self.algorithm} model...")
        
        if self.algorithm == 'prc':
            # 初始化PRC水印处理器
            self.watermark_processor = PRCWatermark(
                n=self.config.get('n', 4 * 64 * 64),
                false_positive_rate=self.config.get('false_positive_rate', 1e-5),
                t=self.config.get('prc_t', 3),
                model_id=self.config.get('model_name', 'stabilityai/stable-diffusion-2-1-base'),
                keys_dir=self.config.get('keys_dir', 'keys')
            )
        else:
            # 其他算法的占位符
            self.watermark_processor = None
            print(f"Algorithm {self.algorithm} not implemented yet")
    
    def embed_watermark(self, image_input: Union[str, Image.Image], 
                       watermark_key: str = None, 
                       prompt: str = None,
                       message: str = None,
                       **kwargs) -> Image.Image:
        """
        在图像中嵌入水印
        
        Args:
            image_input: 输入图像（文件路径或PIL Image对象，PRC算法中此参数被忽略）
            watermark_key: 水印密钥ID
            prompt: 生成提示词（用于扩散模型）
            message: 要嵌入的消息（字符串）
            
        Returns:
            含水印的图像
        """
        if self.algorithm == 'prc':
            if self.watermark_processor is None:
                raise RuntimeError("PRC watermark processor not initialized")
            
            if prompt is None:
                raise ValueError("PRC algorithm requires a prompt for image generation")
                
            key_id = watermark_key or self.config.get('watermark_key', 'default')
            
            return self.watermark_processor.embed(
                prompt=prompt,
                message=message,
                key_id=key_id,
                **kwargs
            )
        else:
            # 其他算法的占位符实现
            print(f"Embedding watermark in image using {self.algorithm}")
            
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = image_input
                
            return image
    
    def extract_watermark(self, image_input: Union[str, Image.Image], 
                         watermark_key: str = None,
                         **kwargs) -> Dict[str, Any]:
        """
        从图像中提取水印
        
        Args:
            image_input: 含水印的图像（文件路径或PIL Image对象）
            watermark_key: 水印密钥ID
            
        Returns:
            提取结果，包含是否检测到水印等信息
        """
        if self.algorithm == 'prc':
            if self.watermark_processor is None:
                raise RuntimeError("PRC watermark processor not initialized")
                
            key_id = watermark_key or self.config.get('watermark_key', 'default')
            
            return self.watermark_processor.extract(
                image=image_input,
                key_id=key_id,
                **kwargs
            )
        else:
            # 其他算法的占位符实现
            print(f"Extracting watermark from image using {self.algorithm}")
            return {
                'detected': False,
                'confidence': 0.0,
                'watermark_info': None
            }
    
    def generate_with_watermark(self, prompt: str, watermark_key: str = None, 
                               message: str = None, **kwargs) -> Image.Image:
        """
        生成带水印的图像（等同于embed_watermark，但接口更明确）
        
        Args:
            prompt: 生成提示词
            watermark_key: 水印密钥ID
            message: 要嵌入的消息
            
        Returns:
            生成的含水印图像
        """
        return self.embed_watermark(
            image_input=None,
            watermark_key=watermark_key,
            prompt=prompt,
            message=message,
            **kwargs
        )
    
    def batch_embed(self, images: list, watermark_key: str = None) -> list:
        """批量嵌入水印"""
        return [self.embed_watermark(img, watermark_key) for img in images]
    
    def batch_extract(self, images: list, watermark_key: str = None) -> list:
        """批量提取水印"""
        return [self.extract_watermark(img, watermark_key) for img in images] 