#!/usr/bin/env python3
"""
测试不同参数对图像水印检测准确率的影响
对比优化前后的参数效果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from PIL import Image
from unified.unified_engine import create_unified_engine

def test_image_detection_with_params(replicate_values=[16, 32, 64], chunk_size_values=[8, 16, 32]):
    """测试不同参数组合的检测效果"""
    print("🧪 测试图像水印检测参数优化效果")
    print("=" * 60)
    
    # 创建统一引擎
    engine = create_unified_engine()
    
    # 创建测试图像
    test_image = Image.new("RGB", (512, 512), color=(100, 150, 200))
    test_message = "test_optimization_2025"
    
    print(f"📝 测试消息: '{test_message}'")
    print(f"🖼️  测试图像: {test_image.size} RGB")
    print()
    
    try:
        # 首先嵌入水印
        print("🔨 嵌入水印...")
        watermarked_image = engine.embed("test image", test_message, 'image', image_input=test_image)
        print("✅ 水印嵌入完成")
        print()
        
        # 测试不同参数组合
        results = []
        
        for replicate in replicate_values:
            for chunk_size in chunk_size_values:
                print(f"🔍 测试参数: replicate={replicate}, chunk_size={chunk_size}")
                
                try:
                    result = engine.extract(watermarked_image, 'image', 
                                          replicate=replicate, chunk_size=chunk_size)
                    
                    detected = result.get('detected', False)
                    confidence = result.get('confidence', 0.0)
                    message = result.get('message', '')
                    message_match = message == test_message
                    
                    print(f"   检测: {'✅' if detected else '❌'} | "
                          f"置信度: {confidence:.3f} | "
                          f"消息匹配: {'✅' if message_match else '❌'} | "
                          f"消息: '{message}'")
                    
                    results.append({
                        'replicate': replicate,
                        'chunk_size': chunk_size,
                        'detected': detected,
                        'confidence': confidence,
                        'message_match': message_match,
                        'message': message
                    })
                    
                except Exception as e:
                    print(f"   ❌ 错误: {e}")
                    results.append({
                        'replicate': replicate,
                        'chunk_size': chunk_size,
                        'detected': False,
                        'confidence': 0.0,
                        'message_match': False,
                        'error': str(e)
                    })
                
                print()
        
        # 分析结果
        print("📊 结果分析")
        print("=" * 60)
        
        successful_results = [r for r in results if r.get('detected') or r.get('confidence', 0) > 0.05]
        
        if successful_results:
            # 按置信度排序
            successful_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            print("🏆 最佳参数组合（按置信度排序）:")
            for i, r in enumerate(successful_results[:5], 1):
                status = "✅ 完美" if (r.get('detected') and r.get('message_match')) else "🔶 部分"
                print(f"  {i}. replicate={r['replicate']}, chunk_size={r['chunk_size']} | "
                      f"置信度: {r.get('confidence', 0):.3f} | {status}")
            
            best_result = successful_results[0]
            print(f"\n🎯 推荐参数: replicate={best_result['replicate']}, chunk_size={best_result['chunk_size']}")
            
        else:
            print("❌ 没有成功的检测结果")
            
        # 显示优化后默认参数的效果
        print(f"\n🔧 当前优化后默认参数效果:")
        default_result = engine.extract(watermarked_image, 'image')  # 使用默认参数
        print(f"   检测: {'✅' if default_result.get('detected') else '❌'} | "
              f"置信度: {default_result.get('confidence', 0):.3f} | "
              f"消息: '{default_result.get('message', '')}'")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_default_params():
    """测试优化后的默认参数"""
    print("🧪 测试优化后的默认参数效果")
    print("=" * 60)
    
    engine = create_unified_engine()
    
    # 创建测试图像
    test_image = Image.new("RGB", (256, 256), color=(255, 128, 64))
    test_message = "default_params_test"
    
    try:
        print("🔨 嵌入水印...")
        watermarked_image = engine.embed("test", test_message, 'image', image_input=test_image)
        print("✅ 嵌入完成")
        
        print("🔍 使用默认参数提取...")
        result = engine.extract(watermarked_image, 'image')
        
        print(f"📋 结果:")
        print(f"   检测到水印: {'✅' if result.get('detected') else '❌'}")
        print(f"   置信度: {result.get('confidence', 0):.3f}")
        print(f"   提取消息: '{result.get('message', '')}'")
        print(f"   消息匹配: {'✅' if result.get('message') == test_message else '❌'}")
        
        success = result.get('detected') or result.get('confidence', 0) > 0.05
        print(f"\n🎯 结果: {'✅ 成功' if success else '❌ 失败'}")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试图像水印检测参数优化")
    parser.add_argument('--mode', choices=['default', 'compare', 'both'], default='both',
                       help='测试模式: default(仅默认参数), compare(参数对比), both(两者)')
    parser.add_argument('--replicate', nargs='+', type=int, default=[16, 32, 64],
                       help='要测试的replicate值')
    parser.add_argument('--chunk-size', nargs='+', type=int, default=[8, 16, 32],
                       help='要测试的chunk_size值')
    
    args = parser.parse_args()
    
    if args.mode in ('default', 'both'):
        success = test_default_params()
        if args.mode == 'default':
            sys.exit(0 if success else 1)
        print("\n")
    
    if args.mode in ('compare', 'both'):
        test_image_detection_with_params(args.replicate, getattr(args, 'chunk_size'))
    
    print("\n✅ 测试完成")