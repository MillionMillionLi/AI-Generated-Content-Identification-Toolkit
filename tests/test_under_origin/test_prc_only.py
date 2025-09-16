#!/usr/bin/env python3
"""
PRC水印专用测试脚本 - 只测试图像水印功能，不涉及文本水印
"""

import os
import sys
import traceback
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_direct_prc_import():
    """测试1: 直接导入PRC水印类"""
    print("=== 测试1: 直接导入PRC水印类 ===")
    try:
        # 直接添加image_watermark路径，避免经过src.__init__.py
        image_watermark_path = os.path.join(project_root, 'src', 'image_watermark')
        sys.path.insert(0, image_watermark_path)
        
        from prc_watermark import PRCWatermark
        print("✓ PRCWatermark类导入成功")
        return True, PRCWatermark
    except Exception as e:
        print(f"✗ PRCWatermark类导入失败: {e}")
        traceback.print_exc()
        return False, None

def test_prc_initialization(PRCWatermark):
    """测试2: PRC水印初始化"""
    print("\n=== 测试2: PRC水印初始化 ===")
    try:
        # 使用测试目录避免污染
        prc = PRCWatermark(
            keys_dir="test_prc_keys",
            cache_dir="/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"
        )
        print("✓ PRCWatermark实例创建成功")
        return True, prc
    except Exception as e:
        print(f"✗ PRCWatermark初始化失败: {e}")
        traceback.print_exc()
        return False, None

def test_key_generation(prc):
    """测试3: 密钥生成功能"""
    print("\n=== 测试3: 密钥生成功能 ===")
    try:
        key_file = prc.generate_key("test_key", message_length=256)
        print(f"✓ 密钥生成成功: {key_file}")
        
        # 验证文件存在
        if os.path.exists(key_file):
            print("✓ 密钥文件已创建")
            return True
        else:
            print("✗ 密钥文件未找到")
            return False
    except Exception as e:
        print(f"✗ 密钥生成失败: {e}")
        traceback.print_exc()
        return False

def test_model_loading(prc):
    """测试4: Stable Diffusion模型加载"""
    print("\n=== 测试4: Stable Diffusion模型加载 ===")
    try:
        print("正在加载Stable Diffusion模型...")
        print("注意: 首次加载可能需要几分钟时间")
        start_time = time.time()
        
        # 触发模型加载
        prc._setup_diffusion_pipe()
        
        load_time = time.time() - start_time
        print(f"✓ 模型加载成功，耗时: {load_time:.2f}秒")
        print(f"✓ 管道类型: {type(prc.pipe)}")
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        traceback.print_exc()
        return False

def test_prc_core_algorithms(prc):
    """测试5: PRC核心算法功能"""
    print("\n=== 测试5: PRC核心算法功能 ===")
    try:
        # 获取密钥
        encoding_key, decoding_key = prc._get_or_create_keys("test_core", 128)
        print("✓ PRC密钥获取成功")
        
        # 测试消息编码
        from prc_watermark import str_to_bin, bin_to_str
        test_message = "Hello PRC!"
        message_bits = str_to_bin(test_message)
        print(f"✓ 消息编码成功: '{test_message}' -> {len(message_bits)} bits")
        
        # 测试消息解码
        decoded_message = bin_to_str(message_bits)
        print(f"✓ 消息解码成功: {len(message_bits)} bits -> '{decoded_message}'")
        
        if decoded_message == test_message:
            print("✓ 消息编解码一致性验证通过")
            return True
        else:
            print(f"✗ 消息编解码不一致: '{test_message}' != '{decoded_message}'")
            return False
            
    except Exception as e:
        print(f"✗ PRC核心算法测试失败: {e}")
        traceback.print_exc()
        return False

def test_embed_functionality(prc):
    """测试6: 水印嵌入功能"""
    print("\n=== 测试6: 水印嵌入功能 ===")
    try:
        prompt = "A beautiful sunset over the ocean"
        message = "PRC Test"
        
        print(f"提示词: {prompt}")
        print(f"嵌入消息: {message}")
        print("正在生成带水印的图像...")
        print("注意: 图像生成可能需要几分钟时间")
        
        start_time = time.time()
        
        watermarked_image = prc.embed(
            prompt=prompt,
            message=message,
            key_id="test_embed",
            num_inference_steps=10,  # 使用较少步数以加快测试
            seed=42
        )
        
        generation_time = time.time() - start_time
        print(f"✓ 图像生成成功，耗时: {generation_time:.2f}秒")
        print(f"✓ 图像类型: {type(watermarked_image)}")
        print(f"✓ 图像尺寸: {watermarked_image.size}")
        
        # 保存图像
        output_dir = "test_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, "prc_test_watermarked.png")
        watermarked_image.save(output_path)
        print(f"✓ 图像已保存到: {output_path}")
        
        return True, watermarked_image
        
    except Exception as e:
        print(f"✗ 水印嵌入失败: {e}")
        traceback.print_exc()
        return False, None

def test_modes_comparison(prc, watermarked_image):
    """测试8: 不同模式对比"""
    print("\n=== 测试8: 不同模式对比 ===")
    
    modes = ['fast', 'accurate', 'exact']
    results = {}
    
    for mode in modes:
        print(f"\n--- 测试 {mode.upper()} 模式 ---")
        try:
            start_time = time.time()
            result = prc.extract(
                image=watermarked_image,
                key_id="test_embed", 
                mode=mode
            )
            extraction_time = time.time() - start_time
            
            results[mode] = {
                'success': True,
                'time': extraction_time,
                'detected': result['detected'],
                'confidence': result['confidence'],
                'message': result.get('message'),
                'mode_used': result.get('mode_used')
            }
            
            print(f"✓ {mode}模式完成，耗时: {extraction_time:.2f}秒")
            print(f"检测结果: {result['detected']}")
            print(f"置信度: {result['confidence']}")
            print(f"解码消息: {result['message']}")
            
        except Exception as e:
            extraction_time = time.time() - start_time if 'start_time' in locals() else 0
            results[mode] = {
                'success': False,
                'time': extraction_time,
                'error': str(e)
            }
            print(f"✗ {mode}模式失败: {e}")
    
    # 结果总结
    print(f"\n--- 模式对比结果 ---")
    for mode, result in results.items():
        if result['success']:
            status = "✓ 检测成功" if result['detected'] else "✗ 未检测到"
            print(f"{mode.upper():>8}: {status} | 耗时: {result['time']:.2f}s")
        else:
            print(f"{mode.upper():>8}: ✗ 运行失败 | 耗时: {result['time']:.2f}s")
    
    return len([r for r in results.values() if r['success']]) > 0



def main():
    """主测试函数"""
    print("PRC水印专用测试")
    print("=" * 50)
    print("测试环境:")
    print(f"- Python版本: {sys.version}")
    print(f"- 工作目录: {os.getcwd()}")
    print(f"- 模型路径: /fs-computility/wangxuhong/limeilin/.cache/huggingface/hub")
    print("=" * 50)
    
    tests_results = []
    prc = None
    watermarked_image = None
    
    # 测试1: 导入
    success, PRCWatermark = test_direct_prc_import()
    tests_results.append(("导入PRC类", success))
    if not success:
        print("\n❌ 基础导入失败，无法继续测试")
        return
    
    # 测试2: 初始化
    success, prc = test_prc_initialization(PRCWatermark)
    tests_results.append(("PRC初始化", success))
    if not success:
        print("\n❌ 初始化失败，无法继续测试")
        return
    
    # 测试3: 密钥生成
    success = test_key_generation(prc)
    tests_results.append(("密钥生成", success))
    
    # 测试4: 模型加载
    success = test_model_loading(prc)
    tests_results.append(("模型加载", success))
    if not success:
        print("\n❌ 模型加载失败，无法进行图像生成测试")
        print_summary(tests_results)
        return
    
    # 测试5: PRC算法
    success = test_prc_core_algorithms(prc)
    tests_results.append(("PRC算法", success))
    
    # 测试6: 水印嵌入
    success, watermarked_image = test_embed_functionality(prc)
    tests_results.append(("水印嵌入", success))
    if not success:
        print("\n❌ 水印嵌入失败，无法进行提取测试")
        print_summary(tests_results)
        return
    
    
    # 测试8: 模式对比
    success = test_modes_comparison(prc, watermarked_image)
    tests_results.append(("模式对比", success))
    
    print_summary(tests_results)

def print_summary(results):
    """打印测试总结"""
    print("\n" + "=" * 50)
    print("测试总结:")
    print("-" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:15} : {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！PRC水印系统工作正常")
    elif passed >= total - 1:
        print("✅ 核心功能正常，个别功能可能需要调优")
    elif passed >= 4:
        print("⚠ 基础功能正常，但完整流程存在问题")
    else:
        print("❌ 系统存在较多问题，需要进一步调试")

if __name__ == "__main__":
    main()