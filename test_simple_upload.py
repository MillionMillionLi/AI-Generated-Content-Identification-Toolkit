#!/usr/bin/env python3
"""
简单测试上传图像文件+水印嵌入功能
"""

import os
import sys
import tempfile
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    from src.unified.watermark_tool import WatermarkTool
    print("✅ 成功导入 WatermarkTool")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def create_test_image(path):
    """创建测试图像"""
    img_array = np.random.rand(256, 256, 3) * 255
    img = Image.fromarray(img_array.astype('uint8'))
    img.save(path)
    print(f"✅ 创建测试图像: {path}")

def test_image_upload_watermark():
    """测试图像上传+水印嵌入"""
    print("🚀 测试图像上传+水印嵌入功能")
    print("=" * 50)
    
    try:
        # 创建WatermarkTool实例
        print("📝 初始化水印工具...")
        tool = WatermarkTool()
        print("✅ 水印工具初始化成功")
        
        # 创建测试图像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image_path = tmp_file.name
        
        create_test_image(test_image_path)
        
        try:
            # 测试图像上传+水印嵌入
            print("🖼️ 开始图像水印嵌入...")
            result = tool.embed(
                prompt="uploaded image",  # 提示词 
                message="test_upload_watermark",  # 水印消息
                modality='image',  # 图像模态
                image_input=test_image_path  # 上传的图像文件路径
            )
            
            print(f"✅ 图像水印嵌入成功!")
            print(f"   结果类型: {type(result)}")
            
            # 保存结果
            if hasattr(result, 'save'):
                output_path = "test_output_image.png"
                result.save(output_path)
                print(f"   结果已保存到: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 图像水印嵌入失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # 清理临时文件
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_image_upload_watermark()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试通过!")
        sys.exit(0)
    else:
        print("💥 测试失败!")
        sys.exit(1)