"""
统一水印工具类 - 提供文本、图像和音频水印的统一接口
"""

import torch
from typing import Dict, Any, Optional, Union
from PIL import Image

from src.text_watermark.credid_watermark import CredIDWatermark
from src.image_watermark.image_watermark import ImageWatermark

# 尝试导入音频水印模块
try:
    from src.audio_watermark.audio_watermark import AudioWatermark
    HAS_AUDIO_WATERMARK = True
except ImportError:
    HAS_AUDIO_WATERMARK = False


class WatermarkTool:
    """统一水印工具类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化统一水印工具
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        # 文本水印当前实现期望直接传入配置字典，这里使用空配置以启用默认参数
        self.text_watermark = CredIDWatermark({})
        self.image_watermark = ImageWatermark(config_path)
        
        # 初始化音频水印处理器（如果可用）
        if HAS_AUDIO_WATERMARK:
            self.audio_watermark = AudioWatermark(config_path)
        else:
            self.audio_watermark = None
    
    # 文本水印接口
    def embed_text_watermark(self, text: str, watermark_key: str = None) -> str:
        """嵌入文本水印"""
        return self.text_watermark.embed_watermark(text, watermark_key)
    
    def extract_text_watermark(self, text: str, watermark_key: str = None) -> Dict[str, Any]:
        """提取文本水印"""
        return self.text_watermark.extract_watermark(text, watermark_key)
    
    def batch_embed_text(self, texts: list, watermark_key: str = None) -> list:
        """批量文本水印嵌入"""
        return self.text_watermark.batch_embed(texts, watermark_key)
    
    def batch_extract_text(self, texts: list, watermark_key: str = None) -> list:
        """批量文本水印提取"""
        return self.text_watermark.batch_extract(texts, watermark_key)
    
    # 图像水印接口
    def embed_image_watermark(self, image_input: Union[str, Image.Image], 
                             watermark_key: str = None, 
                             prompt: str = None,
                             message: str = None,
                             **kwargs) -> Image.Image:
        """嵌入图像水印"""
        return self.image_watermark.embed_watermark(
            image_input, watermark_key, prompt, message, **kwargs
        )
    
    def extract_image_watermark(self, image_input: Union[str, Image.Image], 
                               watermark_key: str = None,
                               **kwargs) -> Dict[str, Any]:
        """提取图像水印"""
        return self.image_watermark.extract_watermark(image_input, watermark_key, **kwargs)
    
    def generate_image_with_watermark(self, prompt: str, watermark_key: str = None, 
                                     message: str = None, **kwargs) -> Image.Image:
        """生成带水印的图像"""
        return self.image_watermark.generate_with_watermark(
            prompt, watermark_key, message, **kwargs
        )
    
    def batch_embed_image(self, images: list, watermark_key: str = None) -> list:
        """批量图像水印嵌入"""
        return self.image_watermark.batch_embed(images, watermark_key)
    
    def batch_extract_image(self, images: list, watermark_key: str = None) -> list:
        """批量图像水印提取"""
        return self.image_watermark.batch_extract(images, watermark_key)
    
    # 音频水印接口
    def embed_audio_watermark(self, 
                             audio_input: Union[str, torch.Tensor], 
                             message: str,
                             output_path: Optional[str] = None,
                             **kwargs) -> Union[torch.Tensor, str]:
        """嵌入音频水印"""
        if not self.audio_watermark:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
        
        return self.audio_watermark.embed_watermark(
            audio_input, message, output_path, **kwargs
        )
    
    def extract_audio_watermark(self, 
                               audio_input: Union[str, torch.Tensor],
                               **kwargs) -> Dict[str, Any]:
        """提取音频水印"""
        if not self.audio_watermark:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
        
        return self.audio_watermark.extract_watermark(audio_input, **kwargs)
    
    def generate_audio_with_watermark(self, 
                                    prompt: str, 
                                    message: str,
                                    output_path: Optional[str] = None,
                                    **kwargs) -> Union[torch.Tensor, str]:
        """生成带水印的音频"""
        if not self.audio_watermark:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal和Bark依赖是否已安装")
        
        return self.audio_watermark.generate_audio_with_watermark(
            prompt, message, output_path, **kwargs
        )
    
    def batch_embed_audio(self, 
                         audio_inputs: list, 
                         messages: list,
                         output_dir: Optional[str] = None,
                         **kwargs) -> list:
        """批量音频水印嵌入"""
        if not self.audio_watermark:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
        
        return self.audio_watermark.batch_embed(
            audio_inputs, messages, output_dir, **kwargs
        )
    
    def batch_extract_audio(self, 
                           audio_inputs: list,
                           **kwargs) -> list:
        """批量音频水印提取"""
        if not self.audio_watermark:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
        
        return self.audio_watermark.batch_extract(audio_inputs, **kwargs)
    
    def evaluate_audio_quality(self,
                              original_audio: Union[str, torch.Tensor],
                              watermarked_audio: Union[str, torch.Tensor]) -> Dict[str, float]:
        """评估音频水印对质量的影响"""
        if not self.audio_watermark:
            raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
        
        return self.audio_watermark.evaluate_quality(original_audio, watermarked_audio)
    
    # 通用接口
    def get_supported_algorithms(self) -> Dict[str, list]:
        """获取支持的算法列表"""
        algorithms = {
            'text': ['credid'],  # TODO: 添加更多文本算法
            'image': ['prc', 'videoseal']
        }
        
        # 添加音频算法（如果可用）
        if self.audio_watermark:
            algorithms['audio'] = ['audioseal']
        
        return algorithms
    
    def set_algorithm(self, modality: str, algorithm: str):
        """设置指定模态的算法"""
        if modality == 'text':
            self.text_watermark.algorithm = algorithm
        elif modality == 'image':
            self.image_watermark.algorithm = algorithm
        elif modality == 'audio':
            if not self.audio_watermark:
                raise RuntimeError("音频水印功能不可用，请检查AudioSeal依赖是否已安装")
            self.audio_watermark.algorithm = algorithm
        else:
            raise ValueError(f"Unsupported modality: {modality}")


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Watermark Tool")
    parser.add_argument("--mode", choices=['text', 'image', 'audio'], required=True, 
                       help="Watermark mode")
    parser.add_argument("--action", choices=['embed', 'extract', 'generate'], required=True,
                       help="Action to perform")
    parser.add_argument("--input", help="Input file or text (not required for generate action)")
    parser.add_argument("--key", default="default", help="Watermark key ID")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--prompt", help="Generation prompt (for image/audio mode)")
    parser.add_argument("--message", help="Message to embed")
    parser.add_argument("--voice", help="Voice preset for audio generation")
    
    args = parser.parse_args()
    
    tool = WatermarkTool()
    
    # TODO: Day 8 实现具体的命令行逻辑
    print(f"Running {args.mode} watermark {args.action} on {args.input}")


if __name__ == "__main__":
    main() 