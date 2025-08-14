#!/usr/bin/env python3
"""
对比分块处理vs非分块处理的准确率测试
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.video_watermark.videoseal_wrapper import VideoSealWrapper
from src.video_watermark.utils import VideoIOUtils

def test_chunk_vs_no_chunk():
    """对比分块处理与非分块处理的效果"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 使用真实的测试视频
    test_video_path = project_root / "src/video_watermark/videoseal/assets/videos/1.mp4"
    
    if not test_video_path.exists():
        print(f"❌ 测试视频不存在: {test_video_path}")
        return
    
    print("📹 加载真实测试视频...")
    video_tensor = VideoIOUtils.read_video_frames(str(test_video_path), max_frames=32)
    print(f"视频形状: {video_tensor.shape}")
    
    # 创建wrapper
    wrapper = VideoSealWrapper()
    
    # 测试消息
    test_message = "comparison_test_2025"
    
    print("\n🔄 嵌入水印...")
    watermarked_video = wrapper.embed_watermark(video_tensor, test_message)
    
    print("\n=== 提取对比测试 ===")
    
    # 测试1：不分块处理
    print("\n--- 测试1：不分块处理 ---")
    result_no_chunk = wrapper.extract_watermark(watermarked_video, chunk_size=999)
    
    print(f"检测结果: {result_no_chunk['detected']}")
    print(f"提取消息: '{result_no_chunk['message']}'")
    print(f"置信度: {result_no_chunk['confidence']:.3f}")
    print(f"验证成功: {result_no_chunk['message'] == test_message}")
    
    # 测试2：分块处理
    print("\n--- 测试2：分块处理(chunk_size=16) ---")
    result_chunk = wrapper.extract_watermark(watermarked_video, chunk_size=16)
    
    print(f"检测结果: {result_chunk['detected']}")
    print(f"提取消息: '{result_chunk['message']}'")
    print(f"置信度: {result_chunk['confidence']:.3f}")
    print(f"验证成功: {result_chunk['message'] == test_message}")
    
    # 汇总对比
    print("\n📊 对比结果:")
    confidence_diff = result_chunk['confidence'] - result_no_chunk['confidence']
    no_chunk_success = result_no_chunk['message'] == test_message
    chunk_success = result_chunk['message'] == test_message
    
    print(f"  置信度变化: {confidence_diff:+.3f}")
    print(f"  非分块验证: {'✅' if no_chunk_success else '❌'}")
    print(f"  分块验证: {'✅' if chunk_success else '❌'}")
    
    if chunk_success and not no_chunk_success:
        print("🎉 分块处理显著改善了提取准确率！")
    elif chunk_success and no_chunk_success:
        print("✅ 两种方法都成功，分块处理保持了稳定性")
    elif not chunk_success and no_chunk_success:
        print("⚠️ 分块处理降低了准确率")
    else:
        print("❌ 两种方法都失败")

if __name__ == "__main__":
    test_chunk_vs_no_chunk()