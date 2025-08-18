#!/usr/bin/env python3
"""
音频水印端到端演示脚本
展示完整的文本→音频→水印→提取流程

使用方法:
    python audio_watermark_demo.py --mode basic    # 基础演示 (仅AudioSeal)
    python audio_watermark_demo.py --mode full     # 完整演示 (AudioSeal + Bark)
    python audio_watermark_demo.py --mode custom   # 自定义演示
"""

import os
import sys
import argparse
import logging
import torch
import time
from pathlib import Path
from typing import Optional, Dict, Any

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.unified.watermark_tool import WatermarkTool
    from src.audio_watermark import (
        AudioWatermark, AUDIOSEAL_AVAILABLE, HAS_BARK,
        AudioIOUtils, AudioVisualizationUtils, print_status
    )
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


class AudioWatermarkDemo:
    """音频水印演示类"""
    
    def __init__(self, output_dir: str = "demo_outputs/audio"):
        """
        初始化演示
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化工具
        self.unified_tool = None
        self.audio_watermark = None
        
        print("🎵 音频水印演示系统")
        print("=" * 50)
    
    def setup_tools(self):
        """初始化工具"""
        self.logger.info("初始化音频水印工具...")
        
        try:
            # 创建统一工具
            self.unified_tool = WatermarkTool()
            
            # 创建专用音频水印工具
            from src.audio_watermark import create_audio_watermark
            self.audio_watermark = create_audio_watermark()
            
            print("✅ 工具初始化成功")
            
        except Exception as e:
            print(f"❌ 工具初始化失败: {e}")
            raise
    
    def print_system_status(self):
        """打印系统状态"""
        print("\n📊 系统状态检查:")
        
        # 检查功能可用性
        if AUDIOSEAL_AVAILABLE:
            print("✅ AudioSeal 音频水印功能可用")
        else:
            print("❌ AudioSeal 不可用 - 请安装AudioSeal")
        
        if HAS_BARK:
            print("✅ Bark 文本转音频功能可用")
        else:
            print("❌ Bark 不可用 - 请安装Bark")
        
        # 检查设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  计算设备: {device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        print()
    
    def create_test_audio(self, duration: float = 3.0) -> torch.Tensor:
        """
        创建测试音频
        
        Args:
            duration: 音频时长(秒)
            
        Returns:
            torch.Tensor: 测试音频
        """
        sample_rate = 16000
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # 创建和弦 (C大调和弦: C-E-G)
        frequencies = [261.63, 329.63, 392.00]  # C4, E4, G4
        audio = torch.zeros_like(t)
        
        for freq in frequencies:
            audio += 0.3 * torch.sin(2 * torch.pi * freq * t)
        
        # 添加包络使声音更自然
        envelope = torch.exp(-t * 0.5)  # 指数衰减
        audio = audio * envelope
        
        return audio.unsqueeze(0)  # 添加通道维度
    
    def demo_basic_watermarking(self):
        """基础水印演示 (仅AudioSeal)"""
        print("🔹 基础音频水印演示")
        print("-" * 30)
        
        if not AUDIOSEAL_AVAILABLE:
            print("❌ AudioSeal不可用，跳过基础演示")
            return
        
        # 创建测试音频
        print("1. 创建测试音频...")
        test_audio = self.create_test_audio(duration=2.0)
        print(f"   音频形状: {test_audio.shape}")
        print(f"   时长: {test_audio.size(-1) / 16000:.2f}秒")
        
        # 保存原始音频
        original_path = self.output_dir / "demo_original.wav"
        AudioIOUtils.save_audio(test_audio, str(original_path), 16000)
        print(f"   原始音频已保存: {original_path}")
        
        # 嵌入水印
        print("\n2. 嵌入水印...")
        test_message = "AudioSeal_Demo_2025"
        print(f"   水印消息: '{test_message}'")
        
        start_time = time.time()
        watermarked_audio = self.audio_watermark.embed_watermark(
            test_audio, test_message
        )
        embed_time = time.time() - start_time
        
        print(f"   嵌入时间: {embed_time:.2f}秒")
        
        # 保存带水印音频
        watermarked_path = self.output_dir / "demo_watermarked.wav"
        AudioIOUtils.save_audio(watermarked_audio, str(watermarked_path), 16000)
        print(f"   带水印音频已保存: {watermarked_path}")
        
        # 提取水印
        print("\n3. 提取水印...")
        start_time = time.time()
        result = self.audio_watermark.extract_watermark(watermarked_audio)
        extract_time = time.time() - start_time
        
        print(f"   提取时间: {extract_time:.2f}秒")
        print(f"   检测结果: {result['detected']}")
        print(f"   置信度: {result['confidence']:.3f}")
        print(f"   提取消息: '{result['message']}'")
        
        # 质量评估
        print("\n4. 质量评估...")
        quality = self.audio_watermark.evaluate_quality(test_audio, watermarked_audio)
        print(f"   信噪比 (SNR): {quality['snr_db']:.2f} dB")
        print(f"   均方误差 (MSE): {quality['mse']:.6f}")
        print(f"   相关性: {quality['correlation']:.4f}")
        
        # 验证
        success = result['detected'] and test_message in result['message']
        print(f"\n5. 验证结果: {'✅ 成功' if success else '❌ 失败'}")
        
        print()
    
    def demo_text_to_audio_watermark(self):
        """文本转音频水印演示 (需要Bark)"""
        print("🔹 文本转音频水印演示")
        print("-" * 30)
        
        if not HAS_BARK:
            print("❌ Bark不可用，跳过文本转音频演示")
            return
        
        if not AUDIOSEAL_AVAILABLE:
            print("❌ AudioSeal不可用，跳过演示")
            return
        
        # 设置文本和消息
        text_prompts = [
            "Hello, this is a demonstration of text to speech with watermark.",
            "你好，这是文本转语音加水印的演示。",
            "Welcome to the audio watermarking system."
        ]
        
        for i, prompt in enumerate(text_prompts):
            print(f"\n{i+1}. 处理文本: '{prompt[:40]}...'")
            
            # 生成带水印的音频
            message = f"demo_message_{i+1}"
            print(f"   水印消息: '{message}'")
            
            try:
                print("   正在生成音频...")
                start_time = time.time()
                
                generated_audio = self.audio_watermark.generate_audio_with_watermark(
                    prompt=prompt,
                    message=message,
                    temperature=0.7,
                    seed=42 + i  # 不同的种子产生不同的语音
                )
                
                generation_time = time.time() - start_time
                print(f"   生成时间: {generation_time:.2f}秒")
                print(f"   音频形状: {generated_audio.shape}")
                print(f"   时长: {generated_audio.size(-1) / 16000:.2f}秒")
                
                # 保存生成的音频
                output_path = self.output_dir / f"demo_generated_{i+1}.wav"
                AudioIOUtils.save_audio(generated_audio, str(output_path), 16000)
                print(f"   音频已保存: {output_path}")
                
                # 验证水印
                print("   验证水印...")
                result = self.audio_watermark.extract_watermark(generated_audio)
                print(f"   检测: {result['detected']}, 置信度: {result['confidence']:.3f}")
                print(f"   消息: '{result['message']}'")
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
        
        print()
    
    def demo_batch_processing(self):
        """批处理演示"""
        print("🔹 批处理演示")
        print("-" * 30)
        
        if not AUDIOSEAL_AVAILABLE:
            print("❌ AudioSeal不可用，跳过批处理演示")
            return
        
        # 创建多个测试音频
        num_audios = 5
        test_audios = []
        test_messages = []
        
        print(f"1. 创建 {num_audios} 个测试音频...")
        for i in range(num_audios):
            # 创建不同频率的测试音频
            duration = 1.0 + i * 0.5  # 不同时长
            audio = self.create_test_audio(duration)
            
            # 添加一些随机变化
            audio = audio + 0.05 * torch.randn_like(audio)
            
            test_audios.append(audio)
            test_messages.append(f"batch_message_{i+1}")
        
        print(f"   创建完成，音频时长范围: {[a.size(-1)/16000 for a in test_audios]}")
        
        # 批量嵌入
        print("\n2. 批量嵌入水印...")
        start_time = time.time()
        watermarked_audios = self.audio_watermark.batch_embed(
            test_audios, test_messages
        )
        batch_embed_time = time.time() - start_time
        
        success_count = sum(1 for a in watermarked_audios if a is not None)
        print(f"   批量嵌入完成: {success_count}/{num_audios} 成功")
        print(f"   总耗时: {batch_embed_time:.2f}秒")
        print(f"   平均耗时: {batch_embed_time/num_audios:.2f}秒/音频")
        
        # 批量提取
        print("\n3. 批量提取水印...")
        start_time = time.time()
        results = self.audio_watermark.batch_extract(watermarked_audios)
        batch_extract_time = time.time() - start_time
        
        print(f"   批量提取完成，总耗时: {batch_extract_time:.2f}秒")
        print(f"   平均耗时: {batch_extract_time/num_audios:.2f}秒/音频")
        
        # 验证结果
        print("\n4. 验证结果:")
        detected_count = 0
        for i, result in enumerate(results):
            detected = result.get('detected', False)
            confidence = result.get('confidence', 0.0)
            message = result.get('message', '')
            
            if detected:
                detected_count += 1
            
            print(f"   音频 {i+1}: {'✅' if detected else '❌'} "
                  f"置信度={confidence:.3f}, 消息='{message}'")
        
        print(f"\n5. 总体结果: {detected_count}/{num_audios} 检测成功 "
              f"({detected_count/num_audios*100:.1f}%)")
        
        print()
    
    def demo_robustness_test(self):
        """鲁棒性测试演示"""
        print("🔹 鲁棒性测试演示")
        print("-" * 30)
        
        if not AUDIOSEAL_AVAILABLE:
            print("❌ AudioSeal不可用，跳过鲁棒性测试")
            return
        
        from src.audio_watermark.utils import AudioProcessingUtils
        
        # 创建测试音频
        test_audio = self.create_test_audio(duration=3.0)
        test_message = "robustness_test_2025"
        
        print("1. 嵌入水印...")
        watermarked_audio = self.audio_watermark.embed_watermark(test_audio, test_message)
        print(f"   原始音频检测基线...")
        
        baseline_result = self.audio_watermark.extract_watermark(watermarked_audio)
        print(f"   基线置信度: {baseline_result['confidence']:.3f}")
        
        # 测试不同的攻击
        print("\n2. 噪声攻击测试:")
        snr_levels = [20, 15, 10, 5, 0]
        
        for snr_db in snr_levels:
            noisy_audio = AudioProcessingUtils.add_noise(
                watermarked_audio, noise_type='white', snr_db=snr_db
            )
            
            result = self.audio_watermark.extract_watermark(noisy_audio)
            status = "✅" if result['detected'] else "❌"
            
            print(f"   SNR {snr_db:2d}dB: {status} 置信度={result['confidence']:.3f}")
        
        print("\n3. 归一化测试:")
        
        # 幅度缩放测试
        scale_factors = [0.1, 0.5, 2.0, 5.0]
        for scale in scale_factors:
            scaled_audio = watermarked_audio * scale
            # 重新归一化到[-1, 1]
            scaled_audio = AudioProcessingUtils.normalize(scaled_audio)
            
            result = self.audio_watermark.extract_watermark(scaled_audio)
            status = "✅" if result['detected'] else "❌"
            
            print(f"   缩放 {scale:3.1f}x: {status} 置信度={result['confidence']:.3f}")
        
        print()
    
    def demo_visualization(self):
        """可视化演示"""
        print("🔹 音频可视化演示")
        print("-" * 30)
        
        try:
            import matplotlib.pyplot as plt
            
            # 创建测试音频
            test_audio = self.create_test_audio(duration=2.0)
            test_message = "visualization_demo"
            
            # 嵌入水印
            watermarked_audio = self.audio_watermark.embed_watermark(test_audio, test_message)
            
            # 生成可视化
            print("1. 生成波形图...")
            waveform_path = self.output_dir / "demo_waveform.png"
            AudioVisualizationUtils.plot_waveform(
                test_audio, 16000, "原始音频波形", str(waveform_path)
            )
            print(f"   波形图已保存: {waveform_path}")
            
            print("2. 生成频谱图...")
            spectrogram_path = self.output_dir / "demo_spectrogram.png"
            AudioVisualizationUtils.plot_spectrogram(
                watermarked_audio, 16000, "带水印音频频谱图", str(spectrogram_path)
            )
            print(f"   频谱图已保存: {spectrogram_path}")
            
        except ImportError:
            print("❌ matplotlib不可用，跳过可视化演示")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
        
        print()
    
    def demo_unified_interface(self):
        """统一接口演示"""
        print("🔹 统一接口演示")
        print("-" * 30)
        
        # 检查支持的算法
        algorithms = self.unified_tool.get_supported_algorithms()
        print("支持的算法:")
        for modality, algs in algorithms.items():
            print(f"  {modality}: {algs}")
        
        if 'audio' not in algorithms:
            print("❌ 统一工具中音频功能不可用")
            return
        
        # 使用统一接口进行演示
        test_audio = self.create_test_audio(duration=1.5)
        test_message = "unified_interface_demo"
        
        print(f"\n使用统一接口处理音频...")
        print(f"水印消息: '{test_message}'")
        
        try:
            # 嵌入
            watermarked = self.unified_tool.embed_audio_watermark(
                test_audio, test_message
            )
            print("✅ 统一接口嵌入成功")
            
            # 提取
            result = self.unified_tool.extract_audio_watermark(watermarked)
            print(f"✅ 统一接口提取成功: {result['detected']}, "
                  f"置信度={result['confidence']:.3f}")
            
            # 质量评估
            quality = self.unified_tool.evaluate_audio_quality(test_audio, watermarked)
            print(f"✅ 质量评估: SNR={quality['snr_db']:.2f}dB")
            
        except Exception as e:
            print(f"❌ 统一接口测试失败: {e}")
        
        print()
    
    def run_demo(self, mode: str = "basic"):
        """
        运行演示
        
        Args:
            mode: 演示模式 ('basic', 'full', 'custom')
        """
        self.print_system_status()
        self.setup_tools()
        
        print(f"📋 演示模式: {mode}")
        print("=" * 50)
        
        if mode == "basic":
            # 基础演示
            self.demo_basic_watermarking()
            self.demo_batch_processing()
            
        elif mode == "full":
            # 完整演示
            self.demo_basic_watermarking()
            self.demo_text_to_audio_watermark()
            self.demo_batch_processing()
            self.demo_robustness_test()
            self.demo_unified_interface()
            self.demo_visualization()
            
        elif mode == "custom":
            # 自定义演示 - 让用户选择
            self._interactive_demo()
        
        print("🎉 演示完成！")
        print(f"📁 输出文件保存在: {self.output_dir}")
    
    def _interactive_demo(self):
        """交互式演示"""
        demos = {
            '1': ('基础水印', self.demo_basic_watermarking),
            '2': ('文本转音频水印', self.demo_text_to_audio_watermark),
            '3': ('批处理', self.demo_batch_processing),
            '4': ('鲁棒性测试', self.demo_robustness_test),
            '5': ('统一接口', self.demo_unified_interface),
            '6': ('可视化', self.demo_visualization),
        }
        
        print("🎯 选择要运行的演示:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  a. 运行全部")
        print("  q. 退出")
        
        while True:
            choice = input("\n请选择 (1-6/a/q): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'a':
                for _, func in demos.values():
                    func()
                break
            elif choice in demos:
                _, func = demos[choice]
                func()
            else:
                print("无效选择，请重试")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="音频水印端到端演示")
    parser.add_argument(
        "--mode", 
        choices=['basic', 'full', 'custom'],
        default='basic',
        help="演示模式"
    )
    parser.add_argument(
        "--output", 
        default="demo_outputs/audio",
        help="输出目录"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行演示
    try:
        demo = AudioWatermarkDemo(args.output)
        demo.run_demo(args.mode)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()