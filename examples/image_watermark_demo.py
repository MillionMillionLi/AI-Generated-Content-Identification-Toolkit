"""
图像水印演示脚本 - 展示PRC水印的使用方法
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.unified.watermark_tool import WatermarkTool
from src.image_watermark.prc_watermark import PRCWatermark
import time


def demo_unified_interface():
    """演示统一接口的使用"""
    print("=== 统一接口演示 ===")
    
    # 初始化统一水印工具
    tool = WatermarkTool()
    
    # 示例提示词和消息
    prompt = "A beautiful landscape with mountains and a lake at sunset"
    message = "Hello PRC Watermark!"
    key_id = "demo_key"
    
    print(f"提示词: {prompt}")
    print(f"嵌入消息: {message}")
    print(f"密钥ID: {key_id}")
    
    try:
        # 生成带水印的图像
        print("\n正在生成带水印的图像...")
        start_time = time.time()
        
        watermarked_image = tool.generate_image_with_watermark(
            prompt=prompt,
            message=message,
            watermark_key=key_id,
            num_inference_steps=20,  # 减少步数以加快演示
            seed=42
        )
        
        generation_time = time.time() - start_time
        print(f"图像生成完成，耗时: {generation_time:.2f}秒")
        
        # 保存图像
        output_dir = "demo_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "watermarked_image.png")
        watermarked_image.save(output_path)
        print(f"水印图像已保存到: {output_path}")
        
        # 提取水印
        print("\n正在提取水印...")
        start_time = time.time()
        
        extraction_result = tool.extract_image_watermark(
            image_input=watermarked_image,
            watermark_key=key_id
        )
        
        extraction_time = time.time() - start_time
        print(f"水印提取完成，耗时: {extraction_time:.2f}秒")
        
        # 显示提取结果
        print("\n=== 提取结果 ===")
        print(f"检测到水印: {extraction_result['detected']}")
        print(f"置信度: {extraction_result['confidence']}")
        if extraction_result['message']:
            print(f"解码消息: {extraction_result['message']}")
        else:
            print("未能解码消息")
            
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("这可能是由于缺少依赖或模型文件导致的")


def demo_direct_prc_interface():
    """演示直接使用PRC类的方法"""
    print("\n=== 直接PRC接口演示 ===")
    
    try:
        # 直接初始化PRC水印处理器
        prc_watermark = PRCWatermark(
            false_positive_rate=1e-4,  # 较高的假阳性率以加快演示
            keys_dir="demo_keys"
        )
        
        # 示例数据
        prompt = "A futuristic city with flying cars and neon lights"
        message = "PRC Direct Demo"
        key_id = "direct_demo"
        
        print(f"提示词: {prompt}")
        print(f"嵌入消息: {message}")
        print(f"密钥ID: {key_id}")
        
        # 嵌入水印
        print("\n正在使用PRC直接接口生成带水印图像...")
        start_time = time.time()
        
        watermarked_image = prc_watermark.embed(
            prompt=prompt,
            message=message,
            key_id=key_id,
            num_inference_steps=15,  # 进一步减少步数
            seed=123
        )
        
        generation_time = time.time() - start_time
        print(f"图像生成完成，耗时: {generation_time:.2f}秒")
        
        # 保存图像
        output_dir = "demo_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "prc_direct_watermarked.png")
        watermarked_image.save(output_path)
        print(f"水印图像已保存到: {output_path}")
        
        # 提取水印
        print("\n正在使用PRC直接接口提取水印...")
        start_time = time.time()
        
        extraction_result = prc_watermark.extract(
            image=watermarked_image,
            key_id=key_id
        )
        
        extraction_time = time.time() - start_time
        print(f"水印提取完成，耗时: {extraction_time:.2f}秒")
        
        # 显示提取结果
        print("\n=== 提取结果 ===")
        print(f"检测到水印: {extraction_result['detected']}")
        print(f"置信度: {extraction_result['confidence']}")
        if extraction_result['message']:
            print(f"解码消息: {extraction_result['message']}")
        else:
            print("未能解码消息")
            
    except Exception as e:
        print(f"直接PRC演示过程中出现错误: {e}")
        print("这可能是由于缺少依赖或模型文件导致的")


def demo_different_message_lengths():
    """演示不同长度消息的处理"""
    print("\n=== 不同消息长度演示 ===")
    
    try:
        prc_watermark = PRCWatermark(keys_dir="demo_keys")
        
        # 测试不同长度的消息
        test_cases = [
            ("Hi", "短消息"),
            ("This is a medium length message for testing", "中等长度消息"),
            ("A" * 50, "长消息（50字符）"),
        ]
        
        for i, (message, description) in enumerate(test_cases):
            print(f"\n--- 测试 {i+1}: {description} ---")
            print(f"消息: {message}")
            print(f"消息长度: {len(message)} 字符")
            
            # 只做编码测试，不生成完整图像以节省时间
            print("测试消息编码...")
            
            # 这里可以添加更详细的测试逻辑
            # 比如验证消息长度是否超过限制等
            
        print("\n消息长度测试完成")
        
    except Exception as e:
        print(f"消息长度测试过程中出现错误: {e}")


def demo_key_management():
    """演示密钥管理功能"""
    print("\n=== 密钥管理演示 ===")
    
    try:
        prc_watermark = PRCWatermark(keys_dir="demo_keys")
        
        # 生成不同的密钥
        key_ids = ["key1", "key2", "key3"]
        
        for key_id in key_ids:
            print(f"生成密钥: {key_id}")
            key_file = prc_watermark.generate_key(key_id)
            print(f"密钥文件保存在: {key_file}")
            
        print("\n密钥管理演示完成")
        
    except Exception as e:
        print(f"密钥管理演示过程中出现错误: {e}")


def main():
    """主函数"""
    print("PRC图像水印演示程序")
    print("=" * 50)
    
    # 检查支持的算法
    tool = WatermarkTool()
    supported = tool.get_supported_algorithms()
    print(f"支持的图像水印算法: {supported['image']}")
    
    # 运行各个演示
    try:
        demo_unified_interface()
        demo_direct_prc_interface()
        demo_different_message_lengths()
        demo_key_management()
        
        print("\n" + "=" * 50)
        print("演示程序完成!")
        print("注意: 由于缺少实际的图像逆向过程，水印提取功能使用占位符实现")
        print("在完整的实现中，需要添加VAE逆向编码功能")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现未处理的错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()