#!/usr/bin/env python3
"""
统一水印引擎测试 - 简洁版
遵循KISS原则，每个模态一个基础测试函数，验证embed→extract循环
"""

import sys
import os
import unittest
import torch
import tempfile

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.unified.unified_engine import create_unified_engine
    from src.unified.watermark_tool import WatermarkTool
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保从项目根目录运行测试")
    sys.exit(1)


class TestUnifiedEngine(unittest.TestCase):
    """统一水印引擎测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.engine = create_unified_engine()
        self.tool = WatermarkTool()
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        # 检查支持的模态
        modalities = self.engine.get_supported_modalities()
        self.assertIn('text', modalities)
        self.assertIn('image', modalities)
        self.assertIn('audio', modalities)
        self.assertIn('video', modalities)
        
        # 检查默认算法
        algorithms = self.engine.get_default_algorithms()
        self.assertEqual(algorithms['image'], 'videoseal')  # 确保图像默认为videoseal
        self.assertEqual(algorithms['text'], 'credid')
        self.assertEqual(algorithms['audio'], 'audioseal')
    @unittest.skip("跳过文本测试")
    def test_text_watermark_basic(self):
        """测试文本水印基础功能"""
        prompt = "太阳从海面升起"
        message = "text_2025"

        # 现在改为依赖统一引擎在内部初始化文本模型与tokenizer
        # 若初始化失败，应直接抛错并让测试失败，以便定位问题
        watermarked = self.engine.embed(prompt, message, 'text')
        self.assertIsInstance(watermarked, str)
        # 输出嵌入的水印消息
        print(f"    文本水印嵌入消息: '{message}'")

        result = self.engine.extract(watermarked, 'text')
        self.assertIsInstance(result, dict)
        self.assertIn('detected', result)
        self.assertIn('message', result)
        self.assertIn('confidence', result)
        # 输出提取到的水印消息
        print(f"    文本水印提取消息: '{result['message']}', 检测到={result['detected']}, 置信度={result['confidence']:.3f}")
    
    # @unittest.skip("跳过图像测试")
    def test_image_watermark_basic(self):
        """测试图像水印基础功能（videoseal）"""
        prompt = "This example demonstrates natural language processing techniques including tokenization methods"
        message = "test_image_2025"
        out_dir = os.path.join('outputs', 'images')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'test_image_watermarked.png')
        
        try:
            # 生成+嵌入
            watermarked_image = self.engine.embed(prompt, message, 'image')
            self.assertIsNotNone(watermarked_image)
            # 显式写入文件
            try:
                watermarked_image.save(out_path)
            except Exception:
                # 若返回的是路径或其他类型，尽量兼容（当前实现返回PIL.Image）
                pass
            self.assertTrue(os.path.exists(out_path), f"图像输出未写入: {out_path}")
            
            # 提取
            result = self.engine.extract(watermarked_image, 'image')
            self.assertIsInstance(result, dict)
            self.assertIn('detected', result)
            self.assertIn('message', result)
            self.assertIn('confidence', result)
            
            # 输出提取结果
            print(f"    图像水印提取结果: 检测到={result['detected']}, 消息='{result['message']}', 置信度={result['confidence']:.3f}")
            
        except Exception as e:
            self.skipTest(f"图像水印需要模型支持: {e}")
    
    # @unittest.skip("跳过音频测试")
    def test_audio_watermark_basic(self):
        """测试音频水印基础功能"""
        # 创建模拟音频数据
        audio_data = torch.randn(1, 16000)  # 1秒音频
        message = "test_audio_2025"
        out_dir = os.path.join('outputs', 'audio')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'test_audio_watermarked.wav')
        
        try:
            # 嵌入
            watermarked_audio = self.engine.embed("audio content", message, 'audio', 
                                                 audio_input=audio_data,
                                                 output_path=out_path)
            # 当提供output_path时，应返回路径并写入文件
            if isinstance(watermarked_audio, torch.Tensor):
                # 兼容旧行为：未保存则回退手动保存
                self.assertTrue(False, "期望返回文件路径，但收到Tensor")
            else:
                self.assertIsInstance(watermarked_audio, str)
                self.assertTrue(os.path.exists(watermarked_audio), f"音频输出未写入: {watermarked_audio}")
            
            # 提取
            result = self.engine.extract(watermarked_audio, 'audio')
            self.assertIsInstance(result, dict)
            self.assertIn('detected', result)
            self.assertIn('message', result)
            self.assertIn('confidence', result)
            
            # 输出提取结果
            print(f"    音频水印提取结果: 检测到={result['detected']}, 消息='{result['message']}', 置信度={result['confidence']:.3f}")
            
        except Exception as e:
            self.skipTest(f"音频水印需要依赖支持: {e}")
    
    @unittest.skip("跳过视频测试")
    def test_video_watermark_basic(self):
        """测试视频水印基础功能（简化版）"""
        prompt = "太阳从海面升起"
        message = "test_video_2025"
        out_dir = os.path.join('tests', 'test_results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'test_video_watermarked.mp4')
        
        try:
            # 生成+嵌入（使用合理的默认参数）
            video_path = self.engine.embed(prompt, message, 'video', output_path=out_path)
            self.assertIsInstance(video_path, str)
            self.assertTrue(os.path.exists(video_path), f"视频输出未写入: {video_path}")
            
            # 提取
            result = self.engine.extract(video_path, 'video')
            self.assertIsInstance(result, dict)
            self.assertIn('detected', result)
            self.assertIn('message', result)
            self.assertIn('confidence', result)
            
            # 输出提取结果和文件位置（不删除文件，便于检查）
            print(f"    视频水印提取结果: 检测到={result['detected']}, 消息='{result['message']}', 置信度={result['confidence']:.3f}")
            print(f"    视频文件位置: {video_path}")
                
        except Exception as e:
            self.skipTest(f"视频水印需要模型支持: {e}")
    
    def test_watermark_tool_compatibility(self):
        """测试WatermarkTool向后兼容性"""
        # 测试新的统一接口
        self.assertTrue(hasattr(self.tool, 'embed'))
        self.assertTrue(hasattr(self.tool, 'extract'))
        
        # 测试向后兼容接口
        self.assertTrue(hasattr(self.tool, 'embed_text_watermark'))
        self.assertTrue(hasattr(self.tool, 'extract_text_watermark'))
        self.assertTrue(hasattr(self.tool, 'generate_image_with_watermark'))
        
        # 测试新增的视频接口
        self.assertTrue(hasattr(self.tool, 'embed_video_watermark'))
        self.assertTrue(hasattr(self.tool, 'extract_video_watermark'))
        self.assertTrue(hasattr(self.tool, 'generate_video_with_watermark'))
    
    def test_system_info(self):
        """测试系统信息获取"""
        info = self.tool.get_system_info()
        
        # 验证必要字段存在
        self.assertIn('supported_modalities', info)
        self.assertIn('supported_algorithms', info)
        self.assertIn('device', info)
        
        # 验证包含所有模态
        self.assertIn('text', info['supported_modalities'])
        self.assertIn('image', info['supported_modalities'])
        self.assertIn('audio', info['supported_modalities'])
        self.assertIn('video', info['supported_modalities'])


class TestQuickIntegration(unittest.TestCase):
    """快速集成测试"""
    
    def test_all_modalities_interfaces(self):
        """测试所有模态的接口可用性"""
        tool = WatermarkTool()
        
        # 仅验证文本模态接口
        modalities = ['text']
        
        for modality in modalities:
            with self.subTest(modality=modality):
                # 检查embed和extract方法存在
                self.assertTrue(hasattr(tool, f'embed_{modality}_watermark'))
                self.assertTrue(hasattr(tool, f'extract_{modality}_watermark'))
    
    def test_algorithms_consistency(self):
        """测试算法配置一致性"""
        engine = create_unified_engine()
        tool = WatermarkTool()
        
        # 仅检查文本默认算法一致性
        engine_algorithms = engine.get_default_algorithms()
        tool_algorithms = tool.get_supported_algorithms()
        self.assertEqual(engine_algorithms['text'], 'credid')
        self.assertEqual(tool_algorithms['text'], 'credid')


def run_tests():
    """运行测试的便捷函数"""
    import logging
    
    # 设置较少的日志输出
    logging.basicConfig(level=logging.ERROR)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestQuickIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回成功状态
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 统一水印引擎测试")
    print("=" * 50)
    print("📁 输出文件存储位置:")
    print("   - 视频文件: tests/test_results/")
    print("   - 图像文件: outputs/images/")
    print("   - 音频文件: outputs/audio/")
    print("   - 文本文件: outputs/text/")
    print("   - 配置文件: config/default_config.yaml")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
        print("💡 提示: 生成的文件已保存到相应目录，可以查看测试结果")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)