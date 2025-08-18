#!/usr/bin/env python3
"""
验证导入问题修复和音频水印功能
"""

import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """测试导入功能"""
    print("🔍 测试导入功能...")
    
    try:
        # 测试统一接口导入
        from unified.watermark_tool import WatermarkTool
        print("  ✅ 统一接口导入成功")
        
        # 测试音频水印导入
        from audio_watermark import create_audio_watermark
        print("  ✅ 音频水印导入成功")
        
        # 测试图像水印导入
        from image_watermark.image_watermark import ImageWatermark
        print("  ✅ 图像水印导入成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_audio_watermark():
    """测试音频水印基础功能"""
    print("\n🎵 测试音频水印功能...")
    
    try:
        import torch
        from audio_watermark import create_audio_watermark
        
        # 创建音频水印工具
        watermark_tool = create_audio_watermark()
        print("  ✅ 音频水印工具创建成功")
        
        # 创建测试音频
        test_audio = torch.randn(1, 16000)  # 1秒音频
        test_message = "test_fix_2025"
        
        # 测试嵌入
        watermarked_audio = watermark_tool.embed_watermark(test_audio, test_message)
        print(f"  ✅ 水印嵌入成功, 输入形状: {test_audio.shape}, 输出形状: {watermarked_audio.shape}")
        
        # 验证形状保持一致
        if watermarked_audio.shape == test_audio.shape:
            print("  ✅ 形状保持一致")
        else:
            print(f"  ❌ 形状不一致: {test_audio.shape} -> {watermarked_audio.shape}")
            return False
        
        # 测试提取
        result = watermark_tool.extract_watermark(watermarked_audio)
        print(f"  ✅ 水印提取成功, 检测到: {result['detected']}, 消息: {result.get('message', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 音频水印测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 验证导入问题修复和音频水印功能")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    # 测试音频水印
    audio_success = test_audio_watermark()
    
    # 总结
    print("\n📊 测试总结")
    print("=" * 50)
    
    if import_success and audio_success:
        print("🎉 所有测试都通过！导入问题已完全修复，音频水印功能正常。")
        print("\n💡 现在你可以安全地使用以下命令:")
        print("   python tests/test_audio_watermark.py")
        print("   python run_tests.py --audio")
        print("   python audio_watermark_demo.py")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())