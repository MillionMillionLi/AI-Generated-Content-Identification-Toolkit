#!/usr/bin/env python3
"""
测试视频下载功能修复
"""

import os
import sys
import requests
import time
import json

# 添加项目路径
sys.path.append('src')

def test_video_download():
    """测试视频生成和下载功能"""
    
    print("🧪 测试视频下载功能修复")
    print("=" * 50)
    
    # 检查服务器状态
    try:
        response = requests.get("http://localhost:5000/")
        if response.status_code != 200:
            print("❌ 服务器未启动，请先运行: python app.py")
            return False
        print("✅ 服务器运行正常")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接服务器，请确保服务器已启动: python app.py")
        return False
    
    # 测试视频生成
    print("\n🎬 测试视频生成和下载...")
    
    video_data = {
        'prompt': '一朵红色的花',
        'message': 'test_download_fix',
        'modality': 'video',
        'num_frames': '16',  # 使用较少帧数加快测试
        'resolution': '320x320'  # 较小分辨率
    }
    
    try:
        # 提交视频生成任务
        print("📤 提交视频生成任务...")
        response = requests.post("http://localhost:5000/api/embed", data=video_data)
        
        if response.status_code != 200:
            print(f"❌ 视频生成失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
        
        result = response.json()
        task_id = result.get('task_id')
        
        if not task_id:
            print("❌ 未获得任务ID")
            return False
            
        print(f"✅ 任务已提交，ID: {task_id}")
        
        # 等待任务完成
        print("⏳ 等待视频生成完成...")
        max_wait = 300  # 5分钟超时
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"http://localhost:5000/api/task/{task_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"进度: {status.get('progress', 0)}% - {status.get('status', 'unknown')}")
                
                if status.get('status') == 'completed':
                    print("✅ 视频生成完成!")
                    break
                elif status.get('status') == 'error':
                    print(f"❌ 视频生成失败: {status.get('error', 'Unknown error')}")
                    return False
            
            time.sleep(5)
        else:
            print("❌ 视频生成超时")
            return False
        
        # 测试文件下载
        print("\n📥 测试文件下载...")
        download_response = requests.get(f"http://localhost:5000/api/download/{task_id}")
        
        if download_response.status_code == 200:
            # 保存下载的文件
            filename = f"test_downloaded_{task_id}.mp4"
            with open(filename, 'wb') as f:
                f.write(download_response.content)
            
            file_size = len(download_response.content)
            print(f"✅ 下载成功! 文件大小: {file_size} 字节")
            print(f"📁 已保存为: {filename}")
            
            # 清理测试文件
            if os.path.exists(filename):
                os.remove(filename)
                print("🧹 已清理测试文件")
            
            return True
        else:
            print(f"❌ 下载失败: {download_response.status_code}")
            print(f"响应: {download_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {e}")
        return False

if __name__ == "__main__":
    success = test_video_download()
    if success:
        print("\n🎉 测试通过! 视频下载功能修复成功!")
    else:
        print("\n💥 测试失败，需要进一步调试")
    
    exit(0 if success else 1)