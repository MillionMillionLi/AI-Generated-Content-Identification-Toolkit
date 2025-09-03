#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
候选消息匹配功能测试脚本
用于验证候选消息保存和匹配功能是否正常工作
"""

import requests
import json
import time
from pathlib import Path

# 服务器配置
BASE_URL = "http://localhost:5000"
TEST_FILES_DIR = Path("test_files")

def test_candidate_matching():
    """测试候选消息匹配功能"""
    print("🚀 开始测试候选消息匹配功能")
    print("=" * 60)
    
    # 测试用例：不同的水印消息
    test_messages = [
        "hello_world_2025",
        "ai_watermark_test",
        "demo_message_123",
        "tech_innovation",
        "secure_watermark"
    ]
    
    # 步骤1：测试多个水印嵌入（建立候选消息库）
    print("\n📝 步骤1：嵌入多个水印消息建立候选库")
    embedded_texts = []
    
    for i, message in enumerate(test_messages):
        print(f"  {i+1}. 嵌入消息: '{message}'")
        
        # 嵌入水印
        embed_data = {
            'modality': 'text',
            'prompt': f'This is test prompt number {i+1}:',
            'message': message
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/embed", data=embed_data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    generated_text = result.get('generated_text', '')
                    embedded_texts.append((message, generated_text))
                    print(f"     ✅ 嵌入成功，生成文本长度: {len(generated_text)}")
                else:
                    print(f"     ❌ 嵌入失败: {result.get('error', 'Unknown error')}")
            else:
                print(f"     ❌ HTTP错误: {response.status_code}")
        except Exception as e:
            print(f"     ❌ 请求异常: {e}")
        
        # 避免请求过于频繁
        time.sleep(2)
    
    print(f"\n成功嵌入 {len(embedded_texts)} 个水印消息")
    
    # 步骤2：查看候选消息库状态
    print("\n📊 步骤2：查看候选消息库状态")
    try:
        response = requests.get(f"{BASE_URL}/api/candidates?modality=text", timeout=30)
        if response.status_code == 200:
            result = response.json()
            stats = result.get('statistics', {})
            candidates = result.get('candidates', {})
            
            print(f"  总候选消息数: {stats.get('total_messages', 0)}")
            print(f"  文本消息数: {stats.get('by_modality', {}).get('text', 0)}")
            print(f"  最近消息数: {stats.get('recent_messages', 0)}")
            print(f"  候选消息列表:")
            
            for msg_id, candidate in candidates.items():
                original = candidate.get('original_message', '')
                binary = candidate.get('encoded_binary', [])
                print(f"    - '{original}' -> {binary} (ID: {msg_id[:8]}...)")
        else:
            print(f"❌ 获取候选消息失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 候选消息查询异常: {e}")
    
    # 步骤3：测试水印提取和匹配
    print("\n🔍 步骤3：测试水印提取和候选消息匹配")
    
    if not embedded_texts:
        print("❌ 没有可用的嵌入文本进行测试")
        return
    
    # 创建测试文件目录
    TEST_FILES_DIR.mkdir(exist_ok=True)
    
    for i, (original_message, watermarked_text) in enumerate(embedded_texts):
        print(f"\n  测试 {i+1}: 原始消息 '{original_message}'")
        
        # 保存到临时文件
        test_file_path = TEST_FILES_DIR / f"test_watermarked_{i+1}.txt"
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(watermarked_text)
        
        # 提取水印
        try:
            with open(test_file_path, 'rb') as f:
                files = {'file': f}
                data = {'modality': 'text'}
                
                response = requests.post(f"{BASE_URL}/api/extract", 
                                       files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                detected = result.get('detected', False)
                extracted_message = result.get('message', '')
                confidence = result.get('confidence', 0.0)
                metadata = result.get('metadata', {})
                
                print(f"    检测结果: {'✅ 检测到' if detected else '❌ 未检测到'}")
                print(f"    提取消息: '{extracted_message}'")
                print(f"    置信度: {confidence:.3f}")
                print(f"    匹配方法: {metadata.get('matching_method', 'standard')}")
                
                # 验证匹配准确性
                if detected and extracted_message == original_message:
                    print(f"    🎯 匹配成功! 完全一致")
                elif detected and extracted_message != original_message:
                    print(f"    ⚠️ 部分匹配: 预期 '{original_message}', 实际 '{extracted_message}'")
                else:
                    print(f"    ❌ 匹配失败")
            else:
                print(f"    ❌ 提取失败: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ 提取异常: {e}")
        
        # 清理测试文件
        try:
            test_file_path.unlink()
        except:
            pass
        
        # 避免请求过于频繁
        time.sleep(1)
    
    # 步骤4：测试匹配阈值
    print("\n🎚️ 步骤4：测试40%匹配阈值")
    print("（这部分需要手动修改水印文本来模拟部分匹配情况）")
    
    print("\n" + "=" * 60)
    print("✅ 候选消息匹配功能测试完成！")
    
    # 清理测试目录
    try:
        if TEST_FILES_DIR.exists():
            import shutil
            shutil.rmtree(TEST_FILES_DIR)
    except:
        pass

def check_server_status():
    """检查服务器状态"""
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 服务器在线: {result.get('tool_status', 'unknown')}")
            return True
        else:
            print(f"❌ 服务器状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        return False

if __name__ == "__main__":
    print("🧪 候选消息匹配功能测试")
    print("=" * 60)
    
    # 检查服务器状态
    if not check_server_status():
        print("\n请确保服务器已启动：python app.py")
        exit(1)
    
    # 运行测试
    try:
        test_candidate_matching()
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()