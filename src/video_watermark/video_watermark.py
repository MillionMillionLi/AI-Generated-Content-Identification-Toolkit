"""
统一视频水印接口
整合HunyuanVideo文生视频和VideoSeal水印技术
"""

import os
import logging
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .model_manager import ModelManager
from .hunyuan_video_generator import HunyuanVideoGenerator
from .videoseal_wrapper import VideoSealWrapper
from .utils import VideoIOUtils, PerformanceTimer, FileUtils, MemoryMonitor


class VideoWatermark:
    """统一视频水印接口类"""
    
    def __init__(
        self, 
        cache_dir: str = "/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub",
        device: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        初始化视频水印工具
        
        Args:
            cache_dir: HuggingFace模型缓存目录
            device: 计算设备 ('cuda', 'cpu', 或None自动选择)
            config: 配置字典，可包含VideoSeal等参数
        """
        self.cache_dir = cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 保存配置
        self.config = config or {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件（延迟加载）
        self.model_manager = None
        self.video_generator = None
        self.watermark_wrapper = None
        
        # 创建缓存目录
        FileUtils.ensure_dir(cache_dir)
        
        self.logger.info(f"VideoWatermark初始化完成，设备: {self.device}")
    
    def _ensure_model_manager(self) -> ModelManager:
        """确保模型管理器已初始化"""
        if self.model_manager is None:
            self.model_manager = ModelManager(self.cache_dir)
        return self.model_manager
    
    def _ensure_video_generator(self) -> HunyuanVideoGenerator:
        """确保视频生成器已初始化"""
        if self.video_generator is None:
            model_manager = self._ensure_model_manager()
            self.video_generator = HunyuanVideoGenerator(model_manager, self.device)
        return self.video_generator
    
    def _ensure_watermark_wrapper(self) -> VideoSealWrapper:
        """确保水印包装器已初始化"""
        if self.watermark_wrapper is None:
            self.watermark_wrapper = VideoSealWrapper(self.device)
        return self.watermark_wrapper
    
    def generate_video_with_watermark(
        self,
        prompt: str,
        message: str,
        output_path: Optional[str] = None,
        # HunyuanVideo参数
        negative_prompt: Optional[str] = None,
        num_frames: int = 49,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
        # VideoSeal参数
        lowres_attenuation: bool = True
    ) -> str:
        """
        文生视频+水印嵌入一体化功能
        
        Args:
            prompt: 文本提示词
            message: 要嵌入的水印消息
            output_path: 输出文件路径，如果None则自动生成
            negative_prompt: 负向提示词
            num_frames: 视频帧数
            height: 视频高度
            width: 视频宽度
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            lowres_attenuation: VideoSeal低分辨率衰减
            
        Returns:
            str: 输出视频文件路径
        """
        self.logger.info("开始文生视频+水印嵌入流程")
        self.logger.info(f"提示词: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        self.logger.info(f"水印消息: '{message}'")
        
        with PerformanceTimer("文生视频+水印嵌入", self.logger):
            # 1. 生成视频tensor
            self.logger.info("步骤1: 生成视频tensor")
            generator = self._ensure_video_generator()
            
            with PerformanceTimer("视频生成", self.logger):
                video_tensor = generator.generate_video_tensor(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            
            self.logger.info(f"视频生成完成: {video_tensor.shape}")
            
            # 2. 嵌入水印
            self.logger.info("步骤2: 嵌入水印")
            wrapper = self._ensure_watermark_wrapper()
            
            with PerformanceTimer("水印嵌入", self.logger):
                watermarked_tensor = wrapper.embed_watermark(
                    video_tensor=video_tensor,
                    message=message,
                    is_video=True,
                    lowres_attenuation=lowres_attenuation
                )
            
            # 3. 保存视频文件
            self.logger.info("步骤3: 保存视频文件")
            
            # 生成输出路径
            if output_path is None:
                # 创建安全的文件名
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_message = "".join(c for c in message[:20] if c.isalnum() or c in ('-', '_')).rstrip()
                filename = f"hunyuan_{safe_prompt}_{safe_message}.mp4".replace(' ', '_')
                output_path = os.path.join("tests/test_results", filename)
            
            # 确保输出目录存在
            FileUtils.ensure_dir(os.path.dirname(output_path))
            
            # 避免文件名冲突
            output_path = FileUtils.get_unique_filename(output_path)
            
            with PerformanceTimer("视频保存", self.logger):
                VideoIOUtils.save_video_tensor(watermarked_tensor, output_path, fps=8)
            
            file_size = FileUtils.get_file_size_mb(output_path)
            self.logger.info(f"视频已保存: {output_path} ({file_size:.1f} MB)")
            
            return output_path
    
    def embed_watermark(
        self,
        video_path: str,
        message: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        lowres_attenuation: bool = True
    ) -> str:
        """
        在现有视频文件中嵌入水印
        
        Args:
            video_path: 输入视频文件路径
            message: 要嵌入的水印消息
            output_path: 输出文件路径，如果None则自动生成
            max_frames: 最大处理帧数限制
            lowres_attenuation: VideoSeal低分辨率衰减
            
        Returns:
            str: 输出视频文件路径
        """
        self.logger.info(f"开始在现有视频中嵌入水印: {video_path}")
        self.logger.info(f"水印消息: '{message}'")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"输入视频文件不存在: {video_path}")
        
        with PerformanceTimer("视频水印嵌入", self.logger):
            # 1. 读取视频
            self.logger.info("步骤1: 读取视频文件")
            with PerformanceTimer("视频读取", self.logger):
                video_tensor = VideoIOUtils.read_video_frames(video_path, max_frames)
            
            self.logger.info(f"视频读取完成: {video_tensor.shape}")
            
            # 2. 嵌入水印
            self.logger.info("步骤2: 嵌入水印")
            wrapper = self._ensure_watermark_wrapper()
            
            with PerformanceTimer("水印嵌入", self.logger):
                watermarked_tensor = wrapper.embed_watermark(
                    video_tensor=video_tensor,
                    message=message,
                    is_video=True,
                    lowres_attenuation=lowres_attenuation
                )
            
            # 3. 保存视频
            self.logger.info("步骤3: 保存带水印视频")
            
            # 生成输出路径
            if output_path is None:
                input_path = Path(video_path)
                safe_message = "".join(c for c in message[:20] if c.isalnum() or c in ('-', '_')).rstrip()
                output_name = f"{input_path.stem}_watermarked_{safe_message}{input_path.suffix}"
                output_path = os.path.join("tests/test_results", output_name)
            
            # 确保输出目录存在
            FileUtils.ensure_dir(os.path.dirname(output_path))
            
            # 避免文件名冲突
            output_path = FileUtils.get_unique_filename(output_path)
            
            with PerformanceTimer("视频保存", self.logger):
                VideoIOUtils.save_video_tensor(watermarked_tensor, output_path, fps=24)
            
            file_size = FileUtils.get_file_size_mb(output_path)
            self.logger.info(f"带水印视频已保存: {output_path} ({file_size:.1f} MB)")
            
            return output_path
    
    def extract_watermark(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        从视频中提取水印
        
        Args:
            video_path: 带水印的视频文件路径
            max_frames: 最大处理帧数限制
            chunk_size: 分块大小，如果None则从配置读取
            
        Returns:
            Dict[str, Any]: 提取结果，包含detected、message、confidence等字段
        """
        self.logger.info(f"开始从视频中提取水印: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        with PerformanceTimer("水印提取", self.logger):
            # 1. 读取视频
            self.logger.info("步骤1: 读取视频文件")
            with PerformanceTimer("视频读取", self.logger):
                video_tensor = VideoIOUtils.read_video_frames(video_path, max_frames)
            
            self.logger.info(f"视频读取完成: {video_tensor.shape}")
            
            # 2. 提取水印
            self.logger.info("步骤2: 提取水印")
            wrapper = self._ensure_watermark_wrapper()
            
            with PerformanceTimer("水印检测", self.logger):
                # 从配置获取chunk_size，如果没有则使用参数或默认值
                if chunk_size is None:
                    videoseal_config = self.config.get('videoseal', {})
                    watermark_params = videoseal_config.get('watermark_params', {})
                    chunk_size = watermark_params.get('chunk_size', 16)
                
                result = wrapper.extract_watermark(
                    watermarked_video=video_tensor,
                    is_video=True,
                    chunk_size=chunk_size
                )
            
            # 添加额外信息
            result.update({
                "video_path": video_path,
                "video_shape": video_tensor.shape,
                "processing_device": self.device
            })
            
            self.logger.info(
                f"水印提取完成 - 检测: {result['detected']}, "
                f"置信度: {result['confidence']:.3f}, "
                f"消息: '{result['message']}'"
            )
            
            return result
    
    def batch_process_videos(
        self,
        video_paths: list,
        messages: list,
        operation: str = "embed",
        output_dir: str = "tests/test_results",
        **kwargs
    ) -> list:
        """
        批量处理视频
        
        Args:
            video_paths: 视频文件路径列表
            messages: 消息列表（embed操作时使用）
            operation: 操作类型 ('embed' 或 'extract')
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            list: 处理结果列表
        """
        self.logger.info(f"开始批量{operation}操作，处理{len(video_paths)}个视频")
        
        FileUtils.ensure_dir(output_dir)
        results = []
        
        for i, video_path in enumerate(video_paths):
            try:
                self.logger.info(f"处理 {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
                
                if operation == "embed":
                    message = messages[i] if i < len(messages) else f"batch_message_{i+1}"
                    output_path = os.path.join(output_dir, f"batch_watermarked_{i+1}.mp4")
                    
                    result_path = self.embed_watermark(
                        video_path=video_path,
                        message=message,
                        output_path=output_path,
                        **kwargs
                    )
                    
                    results.append({
                        "index": i,
                        "input_path": video_path,
                        "output_path": result_path,
                        "message": message,
                        "success": True
                    })
                
                elif operation == "extract":
                    extract_result = self.extract_watermark(video_path, **kwargs)
                    
                    results.append({
                        "index": i,
                        "input_path": video_path,
                        "extract_result": extract_result,
                        "success": True
                    })
                
                else:
                    raise ValueError(f"不支持的操作类型: {operation}")
            
            except Exception as e:
                self.logger.error(f"处理视频{i+1}失败: {e}")
                results.append({
                    "index": i,
                    "input_path": video_path,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["success"])
        self.logger.info(f"批量处理完成: {success_count}/{len(video_paths)} 成功")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "device": self.device,
            "cache_dir": self.cache_dir,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # GPU内存信息
        if torch.cuda.is_available():
            info["gpu_memory"] = MemoryMonitor.get_gpu_memory_info()
        
        # 模型信息
        if self.model_manager:
            info["hunyuan_model"] = self.model_manager.get_model_info()
        
        if self.video_generator:
            info["video_generator"] = self.video_generator.get_pipeline_info()
        
        if self.watermark_wrapper:
            info["videoseal"] = self.watermark_wrapper.get_model_info()
        
        return info
    
    def clear_cache(self):
        """清理所有缓存以释放内存"""
        self.logger.info("清理所有缓存...")
        
        if self.video_generator:
            self.video_generator.clear_pipeline()
        
        if self.watermark_wrapper:
            self.watermark_wrapper.clear_model()
        
        # 清理GPU缓存
        MemoryMonitor.clear_gpu_cache()
        
        self.logger.info("缓存清理完成")


# 方便的工厂函数
def create_video_watermark(
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> VideoWatermark:
    """
    创建视频水印工具的快捷函数
    
    Args:
        cache_dir: 模型缓存目录
        device: 计算设备
        
    Returns:
        VideoWatermark: 视频水印工具实例
    """
    cache_dir = cache_dir or "/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"
    return VideoWatermark(cache_dir=cache_dir, device=device)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试VideoWatermark统一接口...")
    
    try:
        # 创建视频水印工具
        watermark_tool = create_video_watermark()
        
        # 显示系统信息
        system_info = watermark_tool.get_system_info()
        print("系统信息:")
        for key, value in system_info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # 如果命令行参数包含test，进行简化测试
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("\n开始简化功能测试...")
            
            # 测试1: 文生视频+水印（使用较小参数）
            print("测试1: 文生视频+水印")
            try:
                output_path = watermark_tool.generate_video_with_watermark(
                    prompt="一朵红色的花",
                    message="test_2025",
                    num_frames=16,      # 较少帧数
                    height=320,         # 较小分辨率
                    width=320,
                    num_inference_steps=10,  # 较少步数
                    seed=42
                )
                print(f"✅ 文生视频+水印完成: {output_path}")
                
                # 测试2: 水印提取
                print("测试2: 水印提取")
                extract_result = watermark_tool.extract_watermark(output_path)
                print(f"提取结果: {extract_result}")
                
                # 验证
                success = (extract_result["detected"] and 
                          extract_result["message"] == "test_2025")
                print(f"验证结果: {'✅ 成功' if success else '❌ 失败'}")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()