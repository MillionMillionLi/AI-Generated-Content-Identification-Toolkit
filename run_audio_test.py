#!/usr/bin/env python3
"""
音频水印测试运行脚本
解决导入路径问题的简单脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    """运行音频水印测试"""
    print("🎵 音频水印功能测试")
    print("=" * 50)
    
    try:
        # 检查基础依赖
        import torch
        import torchaudio
        print("✅ PyTorch 可用")
        
        # 检查AudioSeal
        try:
            from src.audio_watermark.audioseal_wrapper import AudioSealWrapper
            print("✅ AudioSeal 可用")
            AUDIOSEAL_AVAILABLE = True
        except ImportError as e:
            print(f"❌ AudioSeal 不可用: {e}")
            AUDIOSEAL_AVAILABLE = False
            
        # 检查Bark
        try:
            from src.audio_watermark.bark_generator import HAS_BARK
            if HAS_BARK:
                print("✅ Bark TTS 可用")
            else:
                print("❌ Bark TTS 不可用")
        except ImportError:
            print("❌ Bark TTS 不可用")
            
        # 检查统一接口
        try:
            from src.unified.watermark_tool import WatermarkTool
            print("✅ 统一水印工具 可用")
        except ImportError as e:
            print(f"❌ 统一水印工具 不可用: {e}")
            
        if not AUDIOSEAL_AVAILABLE:
            print("\n⚠️ AudioSeal不可用，跳过测试")
            return
            
        # 运行基础测试
        print("\n=== 基础功能测试 ===")
        run_basic_test()
        
        # 运行统一接口测试
        print("\n=== 统一接口测试 ===")
        run_unified_test()
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

def run_basic_test():
    """运行基础AudioSeal测试"""
    from src.audio_watermark import create_audio_watermark
    import torch
    import time
    
    # 创建水印工具
    watermark_tool = create_audio_watermark()
    
    # 创建测试音频
    test_audio = torch.randn(1, 16000)  # 1秒音频
    test_message = "test_basic_2025"
    
    print(f"测试音频形状: {test_audio.shape}")
    print(f"测试消息: '{test_message}'")
    
    # 嵌入水印
    start_time = time.time()
    watermarked_audio = watermark_tool.embed_watermark(test_audio, test_message)
    embed_time = time.time() - start_time
    
    print(f"✅ 嵌入完成: {embed_time:.3f}秒")
    
    # 提取水印
    start_time = time.time()
    result = watermark_tool.extract_watermark(watermarked_audio)
    extract_time = time.time() - start_time
    
    print(f"✅ 提取完成: {extract_time:.3f}秒")
    print(f"检测结果: {result['detected']}")
    print(f"解码消息: '{result['message']}'")
    print(f"置信度: {result['confidence']:.3f}")
    
    # 质量评估
    quality = watermark_tool.evaluate_quality(test_audio, watermarked_audio)
    print(f"音频质量 - SNR: {quality['snr_db']:.2f} dB")

def run_unified_test():
    """运行统一接口测试"""
    from src.unified.watermark_tool import WatermarkTool
    import torch
    
    # 创建统一工具
    unified_tool = WatermarkTool()
    
    # 检查音频功能
    algorithms = unified_tool.get_supported_algorithms()
    if 'audio' in algorithms and 'audioseal' in algorithms['audio']:
        print("✅ 统一工具支持音频水印")
        
        # 创建测试音频
        test_audio = torch.randn(1, 16000)
        test_message = "unified_test_2025"
        
        # 测试嵌入
        watermarked = unified_tool.embed_audio_watermark(test_audio, test_message)
        print("✅ 统一接口嵌入成功")
        
        # 测试提取
        result = unified_tool.extract_audio_watermark(watermarked)
        print(f"✅ 统一接口提取: {result['detected']}, 消息: '{result['message']}'")
        
    else:
        print("⚠️ 统一工具中音频功能不可用")

if __name__ == "__main__":
    sys.exit(main())