#!/usr/bin/env python3
"""
统一水印引擎演示 - 简洁版
遵循KISS原则，展示四个模态的基础用法，每个模态不超过10行代码
"""

import sys
import os
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.unified.unified_engine import create_unified_engine

def demo_text_watermark():
    """文本水印演示"""
    print("=== 文本水印演示 ===")
    
    engine = create_unified_engine()
    
    # 使用简单的文本示例
    prompt = "这是一个测试文本。"
    message = "text_demo_2025"
    
    print(f"输入文本: {prompt}")
    print(f"水印消息: {message}")
    
    try:
        # 文本水印需要模型和分词器参数，这里演示接口调用方式
        # watermarked = engine.embed(prompt, message, 'text', model=model, tokenizer=tokenizer)
        # result = engine.extract(watermarked, 'text', model=model, tokenizer=tokenizer)
        print(f"⚠️ 文本水印需要提供model和tokenizer参数")
        print(f"示例调用: engine.embed(prompt, message, 'text', model=model, tokenizer=tokenizer)")
        
    except Exception as e:
        print(f"⚠️ 文本水印需要模型支持: {e}")
    
    print()


def demo_image_watermark():
    """图像水印演示（使用videoseal默认算法）"""
    print("=== 图像水印演示 ===")
    
    engine = create_unified_engine()
    
    prompt = "a beautiful cat"
    message = "image_demo_2025"
    
    print(f"生成提示: {prompt}")
    print(f"水印消息: {message}")
    
    try:
        # 生成+嵌入（使用videoseal默认算法）
        watermarked_image = engine.embed(prompt, message, 'image')
        print(f"✅ 图像水印生成完成: {type(watermarked_image)}")
        
        # 提取
        result = engine.extract(watermarked_image, 'image')
        print(f"📤 提取结果: {result['detected']}, 消息: {result['message']}")
        
    except Exception as e:
        print(f"⚠️ 图像水印演示失败: {e}")
    
    print()


def demo_audio_watermark():
    """音频水印演示"""
    print("=== 音频水印演示 ===")
    
    engine = create_unified_engine()
    
    # 使用基础音频水印（非TTS）
    import torch
    prompt = torch.randn(1, 16000)  # 模拟1秒音频
    message = "audio_demo_2025"
    
    print(f"音频输入: {prompt.shape}")
    print(f"水印消息: {message}")
    
    try:
        # 嵌入
        watermarked_audio = engine.embed("audio content", message, 'audio', audio_input=prompt)
        print(f"✅ 音频水印嵌入完成: {type(watermarked_audio)}")
        
        # 提取
        result = engine.extract(watermarked_audio, 'audio')
        print(f"📤 提取结果: {result['detected']}, 消息: {result['message']}")
        
    except Exception as e:
        print(f"⚠️ 音频水印演示失败: {e}")
    
    print()


def demo_video_watermark():
    """视频水印演示（HunyuanVideo + VideoSeal）"""
    print("=== 视频水印演示 ===")
    
    engine = create_unified_engine()
    
    prompt = "阳光洒在海面上"
    message = "video_demo_2025"
    
    print(f"视频提示: {prompt}")
    print(f"水印消息: {message}")
    
    try:
        # 生成+嵌入（使用较小参数进行演示）
        video_path = engine.embed(prompt, message, 'video', 
                                 num_frames=13, height=320, width=320)
        print(f"✅ 视频水印生成完成: {video_path}")
        
        # 提取
        result = engine.extract(video_path, 'video')
        print(f"📤 提取结果: {result['detected']}, 消息: {result['message']}")
        
    except Exception as e:
        print(f"⚠️ 视频水印演示失败: {e}")
    
    print()


def main():
    """主函数"""
    print("🚀 统一水印引擎演示")
    print("=" * 50)
    
    # 设置简单日志
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出
    
    # 显示支持的功能
    engine = create_unified_engine()
    print(f"支持的模态: {', '.join(engine.get_supported_modalities())}")
    print(f"默认算法: {engine.get_default_algorithms()}")
    print()
    
    # 依次演示四个模态
    demo_text_watermark()
    demo_image_watermark()
    demo_audio_watermark()
    demo_video_watermark()
    
    print("✅ 所有演示完成！")
    print("💡 提示: 实际使用时，部分功能需要相应的模型和依赖")


if __name__ == "__main__":
    main()