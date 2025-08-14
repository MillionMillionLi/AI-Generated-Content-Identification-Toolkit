"""
HunyuanVideo模型下载和缓存管理器
自动处理模型的下载、缓存和加载
"""

import os
import logging
from typing import Optional
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logging.warning("huggingface_hub not available. Model downloading will be disabled.")


class ModelManager:
    """HunyuanVideo模型管理器"""
    
    def __init__(self, cache_dir: str = "/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"):
        """
        初始化模型管理器
        
        Args:
            cache_dir: HuggingFace模型缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HunyuanVideo模型配置
        self.hunyuan_repo = "tencent/HunyuanVideo"
        self.hunyuan_model_dir = self.cache_dir / "models--tencent--HunyuanVideo"
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def _check_local_model_exists(self) -> bool:
        """检查本地是否存在HunyuanVideo模型"""
        # 检查多个可能的路径格式
        possible_paths = [
            self.hunyuan_model_dir,
            self.cache_dir / "tencent--HunyuanVideo",
            self.cache_dir / "hub" / "models--tencent--HunyuanVideo"
        ]
        
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                self.logger.info(f"发现本地HunyuanVideo模型: {path}")
                # 更新实际路径
                self.hunyuan_model_dir = path
                return True
        
        return False
    
    def _find_actual_model_path(self) -> Optional[Path]:
        """查找实际的模型文件路径"""
        if not self.hunyuan_model_dir.exists():
            return None
            
        # 查找模型文件（递归搜索）
        for root, dirs, files in os.walk(self.hunyuan_model_dir):
            # 查找关键文件（如config.json, pytorch_model.bin等）
            key_files = ['config.json', 'model_index.json', 'scheduler']
            if any(f in files or f in dirs for f in key_files):
                return Path(root)
        
        # 如果没有找到关键文件，返回根目录
        if any(self.hunyuan_model_dir.iterdir()):
            return self.hunyuan_model_dir
            
        return None
    
    def ensure_hunyuan_model(self, allow_download: bool = True) -> str:
        """
        确保HunyuanVideo模型可用，如果不存在则下载（可选）
        
        Args:
            allow_download: 是否允许下载模型（默认True）
        
        Returns:
            str: 本地模型路径
            
        Raises:
            RuntimeError: 模型不存在且不允许下载，或下载失败
        """
        # 检查本地模型
        if self._check_local_model_exists():
            actual_path = self._find_actual_model_path()
            if actual_path:
                self.logger.info(f"使用本地HunyuanVideo模型: {actual_path}")
                return str(actual_path)
        
        # 模型不存在
        if not allow_download:
            raise RuntimeError(
                f"HunyuanVideo模型不存在于: {self.cache_dir}\n"
                "请手动下载模型或设置allow_download=True启用自动下载"
            )
        
        # 需要下载模型
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub not available. Please install with: pip install huggingface_hub"
            )
        
        self.logger.info(f"开始下载HunyuanVideo模型到: {self.cache_dir}")
        
        try:
            # 使用snapshot_download下载整个仓库
            downloaded_path = snapshot_download(
                repo_id=self.hunyuan_repo,
                cache_dir=str(self.cache_dir),
                resume_download=True,  # 支持断点续传
                local_files_only=False,
                # force_download=False,  # 不强制重新下载
            )
            
            self.logger.info(f"HunyuanVideo模型下载完成: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            self.logger.error(f"下载HunyuanVideo模型失败: {e}")
            raise RuntimeError(f"Failed to download HunyuanVideo model: {e}")
    
    def get_model_path(self) -> str:
        """
        获取HunyuanVideo模型路径（不触发下载）
        
        Returns:
            str: 模型路径，如果不存在返回空字符串
        """
        if self._check_local_model_exists():
            actual_path = self._find_actual_model_path()
            if actual_path:
                return str(actual_path)
        return ""
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 包含模型信息的字典
        """
        model_path = self.get_model_path()
        
        info = {
            "repo_id": self.hunyuan_repo,
            "cache_dir": str(self.cache_dir),
            "local_path": model_path,
            "exists": bool(model_path),
            "huggingface_hub_available": HF_HUB_AVAILABLE
        }
        
        # 如果模型存在，获取更多信息
        if model_path:
            model_path_obj = Path(model_path)
            info.update({
                "size_mb": sum(f.stat().st_size for f in model_path_obj.rglob('*') if f.is_file()) / (1024*1024),
                "num_files": len([f for f in model_path_obj.rglob('*') if f.is_file()])
            })
        
        return info
    
    def clear_cache(self):
        """清理模型缓存"""
        if self.hunyuan_model_dir.exists():
            import shutil
            shutil.rmtree(self.hunyuan_model_dir)
            self.logger.info(f"已清理模型缓存: {self.hunyuan_model_dir}")


# 方便的工具函数
def get_default_model_manager() -> ModelManager:
    """获取默认的模型管理器实例"""
    return ModelManager()


def ensure_hunyuan_model_available(cache_dir: Optional[str] = None) -> str:
    """
    确保HunyuanVideo模型可用的快捷函数
    
    Args:
        cache_dir: 可选的缓存目录
        
    Returns:
        str: 模型路径
    """
    manager = ModelManager(cache_dir) if cache_dir else get_default_model_manager()
    return manager.ensure_hunyuan_model()


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("测试ModelManager...")
    
    manager = ModelManager()
    
    # 显示模型信息
    info = manager.get_model_info()
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试模型确保功能
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        print("\n开始确保模型可用...")
        try:
            model_path = manager.ensure_hunyuan_model()
            print(f"✅ 模型就绪: {model_path}")
        except Exception as e:
            print(f"❌ 错误: {e}")