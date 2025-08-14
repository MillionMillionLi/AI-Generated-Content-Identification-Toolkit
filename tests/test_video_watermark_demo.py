#!/usr/bin/env python3
"""
HunyuanVideo + VideoSeal 水印功能完整演示程序
展示文生视频+水印嵌入的完整工作流程
"""

import os
import sys
import time
import logging
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入视频水印模块
try:
    from src.video_watermark import VideoWatermark
    from src.video_watermark.utils import PerformanceTimer, FileUtils, MemoryMonitor
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


def load_config():
    """加载配置文件"""
    config_path = project_root / "config" / "video_config.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tests/test_results/demo.log')
        ]
    )


def test_model_download_and_cache():
    """测试1：模型下载和缓存管理"""
    print("=" * 60)
    print("🔧 测试1：模型下载和缓存管理")
    print("=" * 60)
    
    config = load_config()
    cache_dir = config['system']['cache_dir'] if config else "/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"
    
    try:
        with PerformanceTimer("模型管理器初始化"):
            watermark_tool = VideoWatermark(cache_dir=cache_dir)
        
        # 显示系统信息
        print("\n📊 系统信息:")
        system_info = watermark_tool.get_system_info()
        for key, value in system_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 检查HunyuanVideo模型
        print("\n🔍 检查HunyuanVideo模型...")
        model_manager = watermark_tool._ensure_model_manager()
        
        with PerformanceTimer("模型检查/下载"):
            model_path = model_manager.ensure_hunyuan_model()
        
        print(f"✅ HunyuanVideo模型就绪: {model_path}")
        
        # 显示模型信息
        model_info = model_manager.get_model_info()
        print(f"📋 模型信息:")
        print(f"  存在: {model_info['exists']}")
        print(f"  本地路径: {model_info['local_path']}")
        if model_info['exists']:
            print(f"  大小: {model_info.get('size_mb', 0):.1f} MB")
            print(f"  文件数: {model_info.get('num_files', 0)}")
        
        return True, watermark_tool
        
    except Exception as e:
        print(f"❌ 模型下载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_text_to_video_generation(watermark_tool, config):
    """测试2：HunyuanVideo文生视频功能"""
    print("\n" + "=" * 60)
    print("🎬 测试2：HunyuanVideo文生视频功能")
    print("=" * 60)
    
    try:
        # 使用配置文件中的测试参数
        test_params = config['hunyuan_video']['test_params'] if config else {
            'num_frames': 16,
            'height': 320,
            'width': 320,
            'num_inference_steps': 10,
            'fps': 8
        }
        
        test_cases = [
            {
                "prompt": "一朵红色的玫瑰花",
                "description": "简单花朵测试"
            },
            {
                "prompt": "蓝天白云",
                "description": "天空场景测试"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            print(f"\n--- 生成视频 {i+1}: {case['description']} ---")
            print(f"提示词: {case['prompt']}")
            
            try:
                generator = watermark_tool._ensure_video_generator()
                
                with PerformanceTimer(f"视频生成 {i+1}"):
                    # 生成视频tensor（不保存文件）
                    video_tensor = generator.generate_video_tensor(
                        prompt=case['prompt'],
                        seed=42 + i,
                        **test_params
                    )
                
                # 保存为文件用于验证
                output_path = f"tests/test_results/test_generation_{i+1}.mp4"
                FileUtils.ensure_dir(os.path.dirname(output_path))
                
                from src.video_watermark.utils import VideoIOUtils
                VideoIOUtils.save_video_tensor(video_tensor, output_path, fps=test_params.get('fps', 8))
                
                file_size = FileUtils.get_file_size_mb(output_path)
                
                results.append({
                    'case': case,
                    'output_path': output_path,
                    'tensor_shape': video_tensor.shape,
                    'file_size_mb': file_size,
                    'success': True
                })
                
                print(f"✅ 生成完成: {output_path}")
                print(f"📹 视频信息: {video_tensor.shape}, 文件大小: {file_size:.1f} MB")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\n📊 文生视频测试结果: {success_count}/{len(test_cases)} 成功")
        
        return results
        
    except Exception as e:
        print(f"❌ 文生视频测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_video_watermark_integration(watermark_tool, config):
    """测试3：文生视频+水印嵌入完整流程"""
    print("\n" + "=" * 60)
    print("🔐 测试3：文生视频+水印嵌入完整流程")
    print("=" * 60)
    
    try:
        # 使用配置文件中的演示参数
        demo_params = config['demo']['demo_params'] if config else {
            'num_frames': 25,
            'height': 480,
            'width': 640,
            'num_inference_steps': 20,
            'fps': 12
        }
        
        test_cases = config['demo']['test_cases'] if config else [
            {
                "prompt": "一只可爱的小猫在花园里玩耍",
                "message": "demo_cat_2025",
                "description": "小猫演示"
            },
            {
                "prompt": "春天的樱花树下，花瓣飘落",
                "message": "cherry_blossom_scene",
                "description": "樱花场景"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            print(f"\n--- 测试用例 {i+1}: {case['description']} ---")
            print(f"提示词: {case['prompt']}")
            print(f"水印消息: {case['message']}")
            
            try:
                # 完整流程：文生视频 + 水印嵌入
                with PerformanceTimer(f"完整流程 {i+1}"):
                    output_path = watermark_tool.generate_video_with_watermark(
                        prompt=case['prompt'],
                        message=case['message'],
                        seed=100 + i,
                        **demo_params
                    )
                
                file_size = FileUtils.get_file_size_mb(output_path)
                
                print(f"✅ 生成完成: {output_path}")
                print(f"📁 文件大小: {file_size:.1f} MB")
                
                # 验证水印
                print("🔍 验证水印...")
                with PerformanceTimer("水印验证"):
                    extract_result = watermark_tool.extract_watermark(output_path, max_frames=50)
                
                success = (extract_result['detected'] and 
                          extract_result.get('message') == case['message'])
                
                results.append({
                    'case': case,
                    'output_path': output_path,
                    'file_size_mb': file_size,
                    'extract_result': extract_result,
                    'verification_success': success,
                    'success': True
                })
                
                print(f"🔍 水印检测: {'成功' if extract_result['detected'] else '失败'}")
                print(f"📤 提取消息: '{extract_result.get('message', 'None')}'")
                print(f"🎚️ 置信度: {extract_result['confidence']:.3f}")
                print(f"🎯 验证结果: {'✅ 通过' if success else '❌ 失败'}")
                
            except Exception as e:
                print(f"❌ 流程失败: {e}")
                results.append({
                    'case': case,
                    'error': str(e),
                    'success': False
                })
        
        # 汇总结果
        success_count = sum(1 for r in results if r['success'])
        verification_count = sum(1 for r in results if r.get('verification_success', False))
        
        print(f"\n📊 完整流程测试结果:")
        print(f"  生成成功: {success_count}/{len(test_cases)}")
        print(f"  水印验证成功: {verification_count}/{len(test_cases)}")
        
        return results
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_existing_video_watermark(watermark_tool):
    """测试4：现有视频文件水印处理"""
    print("\n" + "=" * 60)
    print("📽️ 测试4：现有视频文件水印处理")
    print("=" * 60)
    
    # 使用VideoSeal自带的测试视频
    test_video = project_root / "src/video_watermark/videoseal/assets/videos/1.mp4"
    
    if not test_video.exists():
        print(f"❌ 测试视频不存在: {test_video}")
        print("跳过现有视频水印测试")
        return []
    
    try:
        test_messages = ["existing_video_test", "batch_process_demo"]
        results = []
        
        for i, message in enumerate(test_messages):
            print(f"\n--- 处理 {i+1}: 消息='{message}' ---")
            
            try:
                # 嵌入水印
                with PerformanceTimer(f"水印嵌入 {i+1}"):
                    watermarked_path = watermark_tool.embed_watermark(
                        video_path=str(test_video),
                        message=message,
                        max_frames=100  # 限制帧数加快测试
                    )
                
                file_size = FileUtils.get_file_size_mb(watermarked_path)
                
                # 提取验证
                with PerformanceTimer(f"水印验证 {i+1}"):
                    extract_result = watermark_tool.extract_watermark(
                        watermarked_path,
                        max_frames=100
                    )
                
                success = (extract_result['detected'] and 
                          extract_result.get('message') == message)
                
                results.append({
                    'message': message,
                    'watermarked_path': watermarked_path,
                    'file_size_mb': file_size,
                    'extract_result': extract_result,
                    'verification_success': success,
                    'success': True
                })
                
                print(f"✅ 嵌入完成: {watermarked_path}")
                print(f"📁 文件大小: {file_size:.1f} MB")
                print(f"🔍 检测结果: {'成功' if extract_result['detected'] else '失败'}")
                print(f"📤 提取消息: '{extract_result.get('message', 'None')}'")
                print(f"🎯 验证: {'✅ 通过' if success else '❌ 失败'}")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                results.append({
                    'message': message,
                    'error': str(e),
                    'success': False
                })
        
        success_count = sum(1 for r in results if r['success'])
        verification_count = sum(1 for r in results if r.get('verification_success', False))
        
        print(f"\n📊 现有视频水印测试结果:")
        print(f"  处理成功: {success_count}/{len(test_messages)}")
        print(f"  水印验证成功: {verification_count}/{len(test_messages)}")
        
        return results
        
    except Exception as e:
        print(f"❌ 现有视频水印测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def display_final_summary(all_results):
    """显示最终汇总报告"""
    print("\n" + "=" * 60)
    print("📋 演示结果汇总报告")
    print("=" * 60)
    
    model_test, generation_results, integration_results, existing_results = all_results
    
    # 模型管理测试
    print(f"🔧 模型管理: {'✅ 成功' if model_test else '❌ 失败'}")
    
    # 文生视频测试
    if generation_results:
        success_count = sum(1 for r in generation_results if r['success'])
        print(f"🎬 文生视频: {success_count}/{len(generation_results)} 成功")
    
    # 完整流程测试
    if integration_results:
        success_count = sum(1 for r in integration_results if r['success'])
        verification_count = sum(1 for r in integration_results if r.get('verification_success', False))
        print(f"🔐 完整流程: {success_count}/{len(integration_results)} 成功")
        print(f"🎯 水印验证: {verification_count}/{len(integration_results)} 通过")
    
    # 现有视频测试
    if existing_results:
        success_count = sum(1 for r in existing_results if r['success'])
        verification_count = sum(1 for r in existing_results if r.get('verification_success', False))
        print(f"📽️ 现有视频: {success_count}/{len(existing_results)} 成功")
        print(f"🎯 水印验证: {verification_count}/{len(existing_results)} 通过")
    
    # 输出文件列表
    print(f"\n📁 生成的文件:")
    results_dir = Path("tests/test_results")
    if results_dir.exists():
        for file_path in sorted(results_dir.glob("*.mp4")):
            size_mb = FileUtils.get_file_size_mb(str(file_path))
            print(f"  - {file_path.name} ({size_mb:.1f} MB)")
    
    # GPU内存使用情况
    gpu_info = MemoryMonitor.get_gpu_memory_info()
    if gpu_info:
        print(f"\n💾 GPU内存使用:")
        for gpu_id, info in gpu_info.items():
            print(f"  {gpu_id}: {info['allocated_gb']:.1f} GB 已分配, {info['cached_gb']:.1f} GB 缓存")


def main():
    """主演示程序"""
    print("🎬 HunyuanVideo + VideoSeal 水印功能完整演示")
    print("=" * 60)
    
    # 设置日志和结果目录
    FileUtils.ensure_dir("tests/test_results")
    setup_logging()
    
    # 加载配置
    config = load_config()
    if not config:
        print("⚠️ 无法加载配置文件，将使用默认参数")
    
    total_start_time = time.time()
    
    try:
        # 测试1：模型下载和缓存
        model_test, watermark_tool = test_model_download_and_cache()
        if not model_test:
            print("❌ 模型管理测试失败，无法继续")
            return
        
        # # 测试2：文生视频功能
        # generation_results = test_text_to_video_generation(watermark_tool, config)
        
        # # 测试3：完整流程测试
        # integration_results = test_video_watermark_integration(watermark_tool, config)
        
        # 测试4：现有视频水印
        existing_results = test_existing_video_watermark(watermark_tool)
        
        # 清理内存
        print("\n🧹 清理内存...")
        watermark_tool.clear_cache()
        
        # # 显示最终汇总
        # all_results = (model_test, generation_results, integration_results, existing_results)
        # display_final_summary(all_results)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - total_start_time
        print(f"\n⏱️ 总演示时间: {total_time:.1f}秒")
        print(f"📁 结果保存在: tests/test_results/")
        print("🎉 演示完成！")


if __name__ == "__main__":
    main()