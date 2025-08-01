#!/usr/bin/env python3
"""
快速开始示例 - 演示统一水印工具的基本用法
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.unified.watermark_tool import WatermarkTool


def test_text_watermark():
    """测试文本水印功能"""
    print("=== 测试文本水印 ===")
    
    # 创建工具实例
    tool = WatermarkTool()
    
    # 测试文本
    original_text = "这是一段需要添加水印的测试文本。"
    watermark_key = "test_key_123"
    
    # 嵌入水印
    print(f"原始文本: {original_text}")
    watermarked_text = tool.embed_text_watermark(original_text, watermark_key)
    print(f"水印文本: {watermarked_text}")
    
    # 提取水印
    result = tool.extract_text_watermark(watermarked_text, watermark_key)
    print(f"提取结果: {result}")
    
    print()


def test_image_watermark():
    """测试图像水印功能"""
    print("=== 测试图像水印 ===")
    
    # 创建工具实例
    tool = WatermarkTool()
    
    watermark_key = "test_key_123"
    
    # 生成带水印的图像
    print("生成带水印的图像...")
    prompt = "一只可爱的猫咪"
    watermarked_image = tool.generate_image_with_watermark(prompt, watermark_key)
    print(f"生成完成，图像尺寸: {watermarked_image.size}")
    
    # 提取水印
    result = tool.extract_image_watermark(watermarked_image, watermark_key)
    print(f"提取结果: {result}")
    
    print()


def test_supported_algorithms():
    """测试支持的算法"""
    print("=== 支持的算法 ===")
    
    tool = WatermarkTool()
    algorithms = tool.get_supported_algorithms()
    
    print(f"文本水印算法: {algorithms['text']}")
    print(f"图像水印算法: {algorithms['image']}")
    
    print()


def main():
    """主函数"""
    print("🚀 统一水印工具 - 快速开始示例")
    print("=" * 50)
    
    try:
        test_supported_algorithms()
        test_text_watermark()
        test_image_watermark()
        
        print("✅ 所有测试完成！")
        print("注意: 当前是Day1的占位符实现，实际水印功能将在后续天数中实现。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 