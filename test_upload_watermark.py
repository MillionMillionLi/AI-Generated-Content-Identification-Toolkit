#!/usr/bin/env python3
"""
测试上传文件+水印嵌入功能的简单脚本
"""

import os
import sys
import requests
import tempfile
from PIL import Image
import numpy as np
import soundfile as sf

def create_test_image(path):
    """创建测试图像"""
    # 创建一个简单的测试图像
    img_array = np.random.rand(256, 256, 3) * 255
    img = Image.fromarray(img_array.astype('uint8'))
    img.save(path)
    print(f"✅ 创建测试图像: {path}")

def create_test_audio(path):
    """创建测试音频"""
    # 创建1秒的正弦波音频
    sample_rate = 16000
    duration = 1.0  # 秒
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    sf.write(path, audio, sample_rate)
    print(f"✅ 创建测试音频: {path}")

def test_upload_watermark(base_url="http://localhost:5000"):
    """测试上传文件+水印嵌入功能"""
    print("🚀 开始测试上传文件+水印嵌入功能")
    print("=" * 50)
    
    # 创建临时文件
    with tempfile.TemporaryDirectory() as temp_dir:
        test_image_path = os.path.join(temp_dir, "test_image.png") 
        test_audio_path = os.path.join(temp_dir, "test_audio.wav")
        
        # 创建测试文件
        create_test_image(test_image_path)
        create_test_audio(test_audio_path)
        
        # 测试API状态
        try:
            print("\n📡 检查API状态...")
            response = requests.get(f"{base_url}/api/status", timeout=10)
            if response.status_code == 200:
                print("✅ API服务运行正常")
            else:
                print(f"❌ API服务异常: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 无法连接到API服务: {e}")
            print("请先启动Web服务器: python app.py")
            return False
        
        # 测试图像上传+水印嵌入
        print("\n🖼️ 测试图像上传+水印嵌入...")
        try:
            with open(test_image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'modality': 'image',
                    'message': 'test_upload_watermark_image',
                    'upload_mode': 'true'
                }
                response = requests.post(f"{base_url}/api/embed", 
                                       files=files, data=data, timeout=60)
                
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 图像水印嵌入成功")
                print(f"   任务ID: {result.get('task_id')}")
                print(f"   输出路径: {result.get('output_path')}")
            else:
                print(f"❌ 图像水印嵌入失败: {response.status_code}")
                print(f"   错误信息: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 图像水印嵌入请求失败: {e}")
            return False
        
        # 测试音频上传+水印嵌入
        print("\n🎵 测试音频上传+水印嵌入...")
        try:
            with open(test_audio_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'modality': 'audio',
                    'message': 'test_upload_watermark_audio',
                    'upload_mode': 'true'
                }
                response = requests.post(f"{base_url}/api/embed", 
                                       files=files, data=data, timeout=60)
                
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 音频水印嵌入成功")
                print(f"   任务ID: {result.get('task_id')}")
                print(f"   输出路径: {result.get('output_path')}")
            else:
                print(f"❌ 音频水印嵌入失败: {response.status_code}")
                print(f"   错误信息: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 音频水印嵌入请求失败: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试完成!")
    return True

if __name__ == '__main__':
    # 检查依赖
    try:
        import requests
        import soundfile as sf
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请安装: pip install requests soundfile")
        sys.exit(1)
    
    success = test_upload_watermark()
    if success:
        print("✅ 测试通过")
        sys.exit(0)
    else:
        print("❌ 测试失败")
        sys.exit(1)