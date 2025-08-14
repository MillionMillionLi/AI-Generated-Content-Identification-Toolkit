#!/usr/bin/env python3
"""
测试没有HunyuanVideo模型时的行为
验证系统能够优雅地处理模型缺失的情况
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_watermark import VideoWatermark
from src.video_watermark.model_manager import ModelManager

def test_model_manager_without_model():
    """测试模型管理器在没有模型时的行为"""
    print("=" * 50)
    print("📋 测试1：模型管理器（无自动下载）")
    print("=" * 50)
    
    try:
        manager = ModelManager()
        
        # 显示模型信息
        info = manager.get_model_info()
        print("模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试不允许下载的情况
        try:
            model_path = manager.ensure_hunyuan_model(allow_download=False)
            print(f"意外成功: {model_path}")
        except RuntimeError as e:
            print(f"✅ 正确捕获错误: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return False


def test_videoseal_only():
    """测试只使用VideoSeal功能（不需要HunyuanVideo）"""
    print("\n" + "=" * 50)
    print("🔐 测试2：纯VideoSeal水印功能")
    print("=" * 50)
    
    try:
        # 直接使用VideoSeal包装器
        from src.video_watermark.videoseal_wrapper import VideoSealWrapper
        from src.video_watermark.utils import create_test_video_tensor
        
        wrapper = VideoSealWrapper()
        
        # 创建测试视频
        test_video = create_test_video_tensor(16, 3, 128, 128, "gradient")
        test_message = "videoseal_only_test"
        
        print(f"测试视频: {test_video.shape}")
        print(f"测试消息: '{test_message}'")
        
        # 嵌入水印
        print("嵌入水印...")
        watermarked_video = wrapper.embed_watermark(test_video, test_message)
        print(f"✅ 水印嵌入完成: {watermarked_video.shape}")
        
        # 提取水印
        print("提取水印...")
        result = wrapper.extract_watermark(watermarked_video)
        
        print(f"检测结果: {result['detected']}")
        print(f"提取消息: '{result['message']}'")
        print(f"置信度: {result['confidence']:.3f}")
        
        success = result['detected'] and result['message'] == test_message
        print(f"验证结果: {'✅ 成功' if success else '❌ 失败'}")
        
        # 清理模型
        wrapper.clear_model()
        
        return success
        
    except Exception as e:
        print(f"❌ VideoSeal测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_video_processing():
    """测试现有视频文件的水印处理"""
    print("\n" + "=" * 50)
    print("📽️ 测试3：现有视频水印处理")
    print("=" * 50)
    
    # 检查VideoSeal测试视频
    test_video = project_root / "src/video_watermark/videoseal/assets/videos/1.mp4"
    
    if not test_video.exists():
        print(f"❌ 测试视频不存在: {test_video}")
        return False
    
    try:
        from src.video_watermark.utils import VideoIOUtils
        from src.video_watermark.videoseal_wrapper import VideoSealWrapper
        
        print(f"使用测试视频: {test_video}")
        
        # 读取视频
        print("读取视频...")
        video_tensor = VideoIOUtils.read_video_frames(str(test_video), max_frames=20)
        print(f"视频读取完成: {video_tensor.shape}")
        
        # 处理水印
        wrapper = VideoSealWrapper()
        test_message = "existing_video_watermark"
        
        print(f"嵌入水印: '{test_message}'")
        watermarked_tensor = wrapper.embed_watermark(video_tensor, test_message)
        
        # 保存处理后的视频
        output_path = "tests/test_results/existing_video_watermarked.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        VideoIOUtils.save_video_tensor(watermarked_tensor, output_path, fps=24)
        
        print(f"保存完成: {output_path}")
        
        # 验证水印
        print("验证水印...")
        result = wrapper.extract_watermark(watermarked_tensor)
        
        success = result['detected'] and result['message'] == test_message
        print(f"检测: {result['detected']}")
        print(f"消息: '{result['message']}'")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"验证: {'✅ 成功' if success else '❌ 失败'}")
        
        # 清理
        wrapper.clear_model()
        
        return success
        
    except Exception as e:
        print(f"❌ 现有视频测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 无HunyuanVideo模型情况下的功能测试")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    results = []
    
    # 测试1：模型管理器
    results.append(("模型管理器", test_model_manager_without_model()))
    
    # 测试2：VideoSeal纯水印功能
    results.append(("VideoSeal水印", test_videoseal_only()))
    
    # 测试3：现有视频处理
    results.append(("现有视频处理", test_existing_video_processing()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    success_count = 0
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总体结果: {success_count}/{len(results)} 通过")
    
    if success_count == len(results):
        print("🎉 所有测试通过！系统可以在没有HunyuanVideo的情况下正常工作")
    else:
        print("⚠️ 部分测试失败，需要检查")
    
    print("\n💡 要使用完整功能，请手动下载HunyuanVideo模型到:")
    print("   /fs-computility/wangxuhong/limeilin/.cache/huggingface/hub/")
    print("   或运行: huggingface-cli download tencent/HunyuanVideo")


if __name__ == "__main__":
    main()