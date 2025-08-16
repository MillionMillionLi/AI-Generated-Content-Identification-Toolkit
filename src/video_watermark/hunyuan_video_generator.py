"""
HunyuanVideo文生视频生成器
基于腾讯HunyuanVideo模型的文本到视频生成功能
"""

import os
import logging
import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path

from .model_manager import ModelManager

# 尝试导入diffusers相关模块
try:
    from diffusers import HunyuanVideoPipeline
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("diffusers not available. Please install with: pip install diffusers")


class HunyuanVideoGenerator:
    """HunyuanVideo文生视频生成器"""
    
    def __init__(self, model_manager: ModelManager, device: Optional[str] = None):
        """
        初始化HunyuanVideo生成器
        
        Args:
            model_manager: 模型管理器实例
            device: 计算设备 ('cuda', 'cpu', 或None自动选择)
        """
        self.model_manager = model_manager
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 检查依赖
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required for HunyuanVideo generation. "
                "Install with: pip install diffusers torch torchvision"
            )
    
    def _load_pipeline(self, allow_download: bool = True):
        """延迟加载HunyuanVideo管道"""
        if self.pipeline is not None:
            return
        
        self.logger.info("正在加载HunyuanVideo管道...")
        
        try:
            # 设置镜像环境变量
            original_endpoint = os.environ.get('HF_ENDPOINT')
            if not original_endpoint:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.logger.info("设置HF_ENDPOINT为镜像站点: https://hf-mirror.com")
            
            # 尝试多个仓库源和网络配置
            repo_candidates = [
                "hunyuanvideo-community/HunyuanVideo",  # 社区diffusers兼容版本
                "tencent/HunyuanVideo"  # 官方版本（回退）
            ]
            
            # 网络配置候选（镜像 -> 直连）
            network_configs = [
                {'HF_ENDPOINT': 'https://hf-mirror.com', 'desc': '镜像站点'},
                {'HF_ENDPOINT': None, 'desc': 'HuggingFace直连'}
            ]
            
            pipeline_loaded = False
            for network_config in network_configs:
                # 设置网络环境
                if network_config['HF_ENDPOINT']:
                    os.environ['HF_ENDPOINT'] = network_config['HF_ENDPOINT']
                else:
                    os.environ.pop('HF_ENDPOINT', None)
                
                self.logger.info(f"尝试网络配置: {network_config['desc']}")
                
                for repo_id in repo_candidates:
                    try:
                        self.logger.info(f"尝试从仓库加载: {repo_id}")
                        self.pipeline = HunyuanVideoPipeline.from_pretrained(
                            repo_id,
                            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                            device_map="balanced" if self.device == 'cuda' else None,
                            cache_dir=str(self.model_manager.cache_dir)
                        )
                        self.logger.info(f"成功从 {repo_id} 加载HunyuanVideo管道 (使用{network_config['desc']})")
                        pipeline_loaded = True
                        break
                    except Exception as e:
                        self.logger.warning(f"从 {repo_id} 加载失败 (使用{network_config['desc']}): {e}")
                        continue
                
                if pipeline_loaded:
                    break
            
            # 恢复原始环境变量
            if original_endpoint:
                os.environ['HF_ENDPOINT'] = original_endpoint
            elif 'HF_ENDPOINT' in os.environ:
                os.environ.pop('HF_ENDPOINT', None)
            
            if not pipeline_loaded:
                raise RuntimeError("所有HunyuanVideo仓库和网络配置都加载失败")
            
            # 移动到指定设备
            if self.device == 'cpu':
                self.pipeline = self.pipeline.to('cpu')
            
            # 启用内存优化
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.logger.info(f"HunyuanVideo管道加载完成，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"加载HunyuanVideo管道失败: {e}")
            raise RuntimeError(f"Failed to load HunyuanVideo pipeline: {e}")
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 49,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Union[torch.Tensor, str]:
        """
        生成视频
        
        Args:
            prompt: 文本提示词
            negative_prompt: 负向提示词
            num_frames: 视频帧数 (默认49帧)
            height: 视频高度 (默认720p)
            width: 视频宽度 (默认1280)
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            output_path: 输出视频路径，如果None则返回tensor
            
        Returns:
            torch.Tensor or str: 视频tensor或输出文件路径
        """
        self._load_pipeline(allow_download=True)  # 允许自动下载
        
        self.logger.info(f"开始生成视频: '{prompt[:50]}...'")
        self.logger.info(f"参数: {num_frames}帧, {height}x{width}, {num_inference_steps}步")
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        try:
            # 生成视频
            with torch.no_grad():
                video_frames = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
                ).frames[0]  # 获取第一个(也是唯一一个)视频
            
            self.logger.info(f"视频生成完成: {video_frames.shape}")
            
            # 如果指定输出路径，保存视频文件
            if output_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 使用diffusers的export_to_video保存
                export_to_video(video_frames, output_path, fps=8)
                
                self.logger.info(f"视频已保存到: {output_path}")
                return output_path
            else:
                # 返回video tensor (numpy array)
                return video_frames
                
        except Exception as e:
            self.logger.error(f"视频生成失败: {e}")
            raise RuntimeError(f"Failed to generate video: {e}")
    
    def generate_video_tensor(
        self,
        prompt: str,
        **kwargs
    ) -> torch.Tensor:
        """
        生成视频tensor (用于后续水印处理)
        
        Args:
            prompt: 文本提示词
            **kwargs: 其他生成参数
            
        Returns:
            torch.Tensor: 视频tensor，形状为 (frames, channels, height, width)
        """
        # 强制不保存文件，只返回tensor
        kwargs['output_path'] = None
        video_frames = self.generate_video(prompt, **kwargs)
        
        # 转换numpy array为torch tensor
        if isinstance(video_frames, np.ndarray):
            # video_frames形状: (frames, height, width, channels)
            # 转换为: (frames, channels, height, width)
            video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float()
            # 归一化到[0, 1]
            video_tensor = video_tensor / 255.0
        else:
            video_tensor = video_frames
        
        return video_tensor
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """获取管道信息"""
        info = {
            "device": self.device,
            "pipeline_loaded": self.pipeline is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE
        }
        
        if self.pipeline is not None:
            info.update({
                "dtype": str(self.pipeline.dtype) if hasattr(self.pipeline, 'dtype') else 'unknown',
                "components": list(self.pipeline.components.keys()) if hasattr(self.pipeline, 'components') else []
            })
        
        return info
    
    def clear_pipeline(self):
        """清理管道以释放内存"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("HunyuanVideo管道已清理")


# 方便的工具函数
def create_hunyuan_generator(
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> HunyuanVideoGenerator:
    """
    创建HunyuanVideo生成器的快捷函数
    
    Args:
        cache_dir: 模型缓存目录
        device: 计算设备
        
    Returns:
        HunyuanVideoGenerator: 生成器实例
    """
    model_manager = ModelManager(cache_dir) if cache_dir else ModelManager()
    return HunyuanVideoGenerator(model_manager, device)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试HunyuanVideoGenerator...")
    
    try:
        generator = create_hunyuan_generator()
        
        # 显示生成器信息
        info = generator.get_pipeline_info()
        print("生成器信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 如果命令行参数包含test，进行实际生成测试
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            print("\n开始生成测试视频...")
            
            # 生成一个简短的测试视频
            test_prompt = "一只可爱的小猫在草地上玩耍"
            
            video_tensor = generator.generate_video_tensor(
                prompt=test_prompt,
                num_frames=25,  # 较短的视频用于测试
                height=480,     # 较低分辨率用于测试
                width=640,
                num_inference_steps=20,  # 较少步数用于快速测试
                seed=42
            )
            
            print(f"✅ 测试视频生成完成: {video_tensor.shape}")
            
            # 也可以保存为文件
            output_path = "test_hunyuan_output.mp4"
            generator.generate_video(
                prompt=test_prompt,
                num_frames=25,
                height=480,
                width=640,
                num_inference_steps=20,
                seed=42,
                output_path=output_path
            )
            
            print(f"✅ 测试视频已保存: {output_path}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()