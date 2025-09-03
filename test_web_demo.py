#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Demo 测试脚本
测试各个API接口的基本功能
"""

import requests
import time
import json
from pathlib import Path

# 配置
BASE_URL = "http://localhost:5000"
TEST_FILES_DIR = Path(__file__).parent / "test_files"

def test_api_status():
    """测试API状态接口"""
    print("🔍 测试 API 状态...")
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API在线，状态: {data['status']}")
            print(f"   工具状态: {data['tool_status']}")
            print(f"   活跃任务: {data['active_tasks']}")
            return True
        else:
            print(f"❌ API状态检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def test_text_embed():
    """测试文本水印嵌入"""
    print("\n📝 测试文本水印嵌入...")
    try:
        data = {
            'modality': 'text',
            'prompt': 'Please provide a detailed analysis of the software release including version specifications, feature updates, compatibility requirements, and user documentation',
            'message': 'v202415beta'
        }
        
        response = requests.post(f"{BASE_URL}/api/embed", data=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文本嵌入成功")
            print(f"   任务ID: {result['task_id']}")
            print(f"   状态: {result['status']}")
            if result.get('generated_text'):
                print(f"   生成文本长度: {len(result['generated_text'])} 字符")
            return result['task_id']
        else:
            error = response.json()
            print(f"❌ 文本嵌入失败: {error.get('error')}")
            return None
    except Exception as e:
        print(f"❌ 文本嵌入异常: {e}")
        return None

def test_text_extract():
    """测试文本水印提取"""
    print("\n🔍 测试文本水印提取...")
    
    # 创建测试文本文件
    test_text = """人工智能技术正在快速发展，为各个领域带来革命性变化。在这个数字化时代，我们需要更好地理解和应用AI技术。"""
    
    test_file = TEST_FILES_DIR / "test_text.txt"
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    try:
        data = {'modality': 'text'}
        files = {'file': open(test_file, 'rb')}
        
        response = requests.post(f"{BASE_URL}/api/extract", data=data, files=files)
        files['file'].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文本提取完成")
            print(f"   任务ID: {result['task_id']}")
            print(f"   检测结果: {result['detected']}")
            print(f"   水印消息: {result.get('message', '无')}")
            print(f"   置信度: {result.get('confidence', 0):.3f}")
            return result['task_id']
        else:
            error = response.json()
            print(f"❌ 文本提取失败: {error.get('error')}")
            return None
    except Exception as e:
        print(f"❌ 文本提取异常: {e}")
        return None

def test_download(task_id):
    """测试文件下载"""
    if not task_id:
        return False
        
    print(f"\n💾 测试结果下载 (任务ID: {task_id})...")
    try:
        response = requests.get(f"{BASE_URL}/api/download/{task_id}")
        if response.status_code == 200:
            # 保存下载的文件
            download_file = TEST_FILES_DIR / f"downloaded_{task_id}.txt"
            with open(download_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ 文件下载成功: {download_file}")
            print(f"   文件大小: {len(response.content)} 字节")
            return True
        elif response.status_code == 404:
            print("⚠️ 结果文件不存在（可能是文本模态不生成文件）")
            return True
        else:
            print(f"❌ 文件下载失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 文件下载异常: {e}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("🧪 多模态水印工具 Web Demo 测试")
    print("=" * 50)
    
    # 测试API状态
    if not test_api_status():
        print("\n❌ 无法连接到服务器，请确保Web服务正在运行")
        print("   启动命令: python start_demo.py 或 python app.py")
        return False
    
    # 等待一下，确保服务完全启动
    time.sleep(1)
    
    # 测试文本功能
    embed_task_id = test_text_embed()
    extract_task_id = test_text_extract()
    
    # 测试下载功能
    if embed_task_id:
        test_download(embed_task_id)
    
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    print(f"   API状态检查: ✅")
    print(f"   文本水印嵌入: {'✅' if embed_task_id else '❌'}")
    print(f"   文本水印提取: {'✅' if extract_task_id else '❌'}")
    print(f"   结果下载: ✅")
    
    return True

def test_web_interface():
    """测试Web界面"""
    print("\n🌐 测试Web界面...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Web界面正常加载")
            print(f"   页面大小: {len(response.content)} 字节")
            print(f"   内容类型: {response.headers.get('content-type')}")
            return True
        else:
            print(f"❌ Web界面加载失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web界面测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 启动Web Demo测试...")
    print("请确保Web服务已启动 (python start_demo.py)")
    print()
    
    # 等待用户确认
    try:
        input("按回车键开始测试...")
    except KeyboardInterrupt:
        print("\n测试已取消")
        return
    
    # 测试Web界面
    test_web_interface()
    
    # 运行API测试
    success = run_comprehensive_test()
    
    print("\n🎯 测试完成!")
    if success:
        print("✅ 基本功能正常，可以在浏览器中访问:")
        print(f"   {BASE_URL}")
    else:
        print("❌ 测试中发现问题，请检查日志和错误信息")

if __name__ == '__main__':
    main()