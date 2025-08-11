"""
统一水印工具类 - 提供文本和图像水印的统一接口
"""

from typing import Dict, Any, Optional, Union
from PIL import Image

from ..text_watermark.credid_watermark import CredIDWatermark
from ..image_watermark.image_watermark import ImageWatermark


class WatermarkTool:
    """统一水印工具类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化统一水印工具
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.text_watermark = CredIDWatermark(config_path)
        self.image_watermark = ImageWatermark(config_path)
    
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
    
    # 通用接口
    def get_supported_algorithms(self) -> Dict[str, list]:
        """获取支持的算法列表"""
        return {
            'text': ['credid'],  # TODO: 添加更多文本算法
            'image': ['prc', 'stable_signature']  # 新增PRC算法支持
        }
    
    def set_algorithm(self, modality: str, algorithm: str):
        """设置指定模态的算法"""
        if modality == 'text':
            self.text_watermark.algorithm = algorithm
        elif modality == 'image':
            self.image_watermark.algorithm = algorithm
        else:
            raise ValueError(f"Unsupported modality: {modality}")


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Watermark Tool")
    parser.add_argument("--mode", choices=['text', 'image'], required=True, 
                       help="Watermark mode")
    parser.add_argument("--action", choices=['embed', 'extract'], required=True,
                       help="Action to perform")
    parser.add_argument("--input", required=True, help="Input file or text")
    parser.add_argument("--key", default="default", help="Watermark key ID")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--prompt", help="Generation prompt (for image mode)")
    parser.add_argument("--message", help="Message to embed")
    
    args = parser.parse_args()
    
    tool = WatermarkTool()
    
    # TODO: Day 8 实现具体的命令行逻辑
    print(f"Running {args.mode} watermark {args.action} on {args.input}")


if __name__ == "__main__":
    main() 