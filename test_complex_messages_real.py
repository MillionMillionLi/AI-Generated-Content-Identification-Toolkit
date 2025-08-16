#!/usr/bin/env python3
"""
水印衰减验证测试
测试循环嵌入vs分段嵌入对水印强度的影响
"""

import os
import sys
import yaml
import torch
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

# 导入水印相关模块
from src.text_watermark.credid_watermark import CredIDWatermark

# 强制离线模式，避免联网
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def _candidate_cache_dirs() -> list:
    """返回可能的本地缓存目录列表（按优先级）。"""
    candidates = []
    # 1) 配置/环境指定
    if os.getenv('HF_HOME'):
        candidates.append(os.path.join(os.getenv('HF_HOME'), 'hub'))
    if os.getenv('HF_HUB_CACHE'):
        candidates.append(os.getenv('HF_HUB_CACHE'))
    # 2) 本项目内 models 目录
    candidates.append(os.path.join(os.path.dirname(__file__), 'models'))
    # 3) 项目上层常见缓存路径
    candidates.append('/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub')
    # 4) 用户主页默认缓存
    candidates.append(os.path.expanduser('~/.cache/huggingface/hub'))
    # 去重并保留顺序
    seen = set()
    ordered = []
    for p in candidates:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered

def load_test_config():
    """加载测试配置"""
    with open('config/text_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 水印衰减测试配置
    config['num_beams'] = 1  # 加速测试
    config['lm_params']['message_len'] = 10  # 10-bit编码
    config['wm_params']['encode_ratio'] = 8   # 降低编码密度，确保足够tokens
    # 调整前缀长度避免跳过第一段  
    if 'lm_prefix_len' in config:
        config['lm_prefix_len'] = 5
    config['max_new_tokens'] = 3000  # 确保足够长的文本生成
    config['confidence_threshold'] = 0.5
    
    return config


def load_model_and_tokenizer(config):
    """加载模型和分词器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = config['model_name']
    print(f"🏗️  加载模型: {model_name}")
    
    start_time = time.time()
    cache_dir = config.get('hf_cache_dir')
    if not cache_dir:
        # 尝试自动发现可用缓存目录
        for c in _candidate_cache_dirs():
            if os.path.isdir(c):
                cache_dir = c
                break
    # 允许 local_files_only，完全离线
    local_only = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_only,
            trust_remote_code=True,
            use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_only,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        tried = [f"model_name='{model_name}'", f"cache_dir='{cache_dir}'"]
        raise RuntimeError(
            "离线加载模型失败。请确认模型已存在于本地缓存，或在config中设置 'hf_cache_dir' 指向本地权重目录。"\
            f" 尝试参数: {', '.join(tried)}\n原始错误: {e}"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_time = time.time() - start_time
    print(f"✅ 模型加载成功 ({load_time:.1f}s)")
    
    return model, tokenizer


def print_detailed_confidence(extract_result, original_segments):
    """打印每段水印的详细置信度信息"""
    print(f"📊 逐段置信度分析:")
    
    if 'detailed_confidence' not in extract_result:
        print("   ⚠️  未找到detailed_confidence信息")
        return
    
    detailed_conf = extract_result['detailed_confidence']
    if not detailed_conf or len(detailed_conf) == 0:
        print("   ⚠️  详细置信度列表为空")
        return
    
    # 直接使用提取的消息进行分析，不需要转换
    extracted_message = extract_result.get('extracted_message', '')
    
    match_count = 0
    total_confidence = 0
    
    # 只显示原始段数的置信度，避免显示冗余的循环段
    display_count = min(len(detailed_conf), len(original_segments))
    
    for i in range(display_count):
        if i < len(detailed_conf) and len(detailed_conf[i]) >= 3:
            conf_data = detailed_conf[i]
            abs_conf = conf_data[0]      # 绝对置信度 
            rel_conf = conf_data[1]      # 相对置信度
            prob_score = conf_data[2]    # 概率分数
            
            # 原始段信息
            orig_seg = original_segments[i] if i < len(original_segments) else "?"
            
            total_confidence += prob_score
            
            print(f"   段{i+1} : 置信度={abs_conf}, 相对={rel_conf}, 概率={prob_score:.3f}")
    
    # 总结统计
    avg_confidence = total_confidence / display_count if display_count > 0 else 0
    print(f"   ✅ 总结: {display_count}段分析完成, 平均概率={avg_confidence:.3f}")
   


def test_cyclic_embedding(message_segments, prompt, config, model, tokenizer):
    """
    测试循环嵌入方法（现有方法）
    
    Args:
        message_segments: 消息段列表 
        prompt: 提示文本
        config: 配置
        model, tokenizer: 模型和分词器
        
    Returns:
        测试结果字典
    """
    print("🔧 方法1: 循环嵌入（现有方法）")
    print("-" * 40)
    
    # 重新组合完整消息
    full_message = "".join(message_segments)
    print(f"🎯 完整消息: '{full_message}'")
    print(f"📝 分段: {message_segments}")
    
    # 使用现有的embed方法（内部循环嵌入）
    watermark = CredIDWatermark(config)
    
    
    start_time = time.time()
    embed_result = watermark.embed(
        model, tokenizer, prompt, message_segments,  # 直接传递段列表
        segmentation_mode="auto"  # 让系统自动识别为字符串列表
    )
    embed_time = time.time() - start_time
    
    if embed_result['success']:
        watermarked_text = embed_result['watermarked_text']
        # 计算token数量用于验证长度充足性
        try:
            # 使用模型自带的tokenizer计算精确token数
            tokens = tokenizer.encode(watermarked_text)
            token_count = len(tokens)
        except:
            # 备用方案：粗略估算 1 token ≈ 4 chars
            token_count = len(watermarked_text) // 4
            
        print(f"✅ 嵌入成功 ({embed_time:.1f}s)")
        print(f"📏 文本长度: {len(watermarked_text)} 字符")
        print(f"🔢 Token估算: ~{token_count} tokens")
        print(f"🎯 理论需求: {len(message_segments)} 段 × {config.get('wm_params', {}).get('encode_ratio', 10) * config.get('lm_params', {}).get('message_len', 10)} = ~{len(message_segments) * config.get('wm_params', {}).get('encode_ratio', 10) * config.get('lm_params', {}).get('message_len', 10)} tokens")
        print(f"📊 长度充足性: {'✅ 充足' if token_count >= len(message_segments) * 100 else '⚠️ 可能不足'}")


        
        # 提取测试
        print(f"\n🔍 开始提取...")
        extract_start = time.time()
        
        extract_result = watermark.extract(
            watermarked_text, model, tokenizer,
            candidates_messages=message_segments  # 传递段列表作为候选
        )
        
        extract_time = time.time() - extract_start
        
        # 显示提取结果详情
        print(f"🔍 提取结果详情:")
        print(f"   提取成功: {'✅' if extract_result['success'] else '❌'}")
        print(f"   提取编码: '{extract_result['binary_message']}'")
        # 使用原始分段计算对应编码，避免与自动分段结果长度不一致导致的越界
        original_binary = watermark._message_to_binary(message_segments)
        print(f"   原始编码: '{original_binary}'")
       
        print(f"   整体置信度: {extract_result['confidence']:.3f}")
        print(f"   提取时间: {extract_time:.1f}s")
       
        
        # 显示逐段详情
        print()
        print_detailed_confidence(extract_result, message_segments)
        
        return {
            'method': 'cyclic',
            'full_message': full_message,
            'segments': message_segments,
            'embed_success': True,
            'embed_time': embed_time,
            'text_length': len(watermarked_text),
            'watermarked_text': watermarked_text,
            'extract_result': extract_result,
            'extract_time': extract_time,    
        }
    else:
        print(f"❌ 嵌入失败: {embed_result.get('error', 'Unknown')}")
        print(f"🎯 原始消息段: {message_segments}")
        return {
            'method': 'cyclic',
            'embed_success': False,
            'error': embed_result.get('error', 'Unknown')
        }


def test_watermark_attenuation():
    
    print("=" * 60)
    
    # 检查CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔥 CUDA: {device_name}")
        print(f"   显存: {memory_gb:.1f} GB")
    else:
        print("⚠️  CUDA不可用，使用CPU")
    
    try:
        # 加载配置和模型
        config = load_test_config()
        print(f"\n📋 测试配置:")
        print(f"   模型: {config['model_name']}")
        print(f"   message_len: {config['lm_params']['message_len']} (10-bit编码)")
        print(f"   encode_ratio: {config['wm_params']['encode_ratio']} ")
        
        print(f"\n🏗️  加载模型...")
        model, tokenizer = load_model_and_tokenizer(config)
        
        # 测试用例 - 每个都有5段消息，使用适中长度的prompt避免对齐问题
        test_cases = [
            {
                'name': '版本号格式',
                'message_segments': ['v', '2024', '1', '5', 'beta'],
                'prompt': 'Please provide a detailed analysis of the software release including version specifications, feature updates, compatibility requirements, and user documentation'
            },
            {
                'name': '文本序列', 
                'message_segments': ['hello', 'world', 'test', 'case', 'one'],
                'prompt': 'This example demonstrates natural language processing techniques including tokenization methods, semantic analysis, machine learning architectures, and practical implementation guidelines'
            },
            {
                'name': '字母序列',
                'message_segments': ['A', 'B', 'C', 'D', 'E'], 
                'prompt': 'The sequence analysis contains algorithmic approaches, data structure implementations, optimization techniques, and performance benchmarking procedures for efficient processing'
            },
            {
                'name': '数字文本混合',
                'message_segments': ['123', 'test', '456', 'demo', '789'],
                'prompt': 'The mixed content analysis shows statistical methodologies, data mining techniques, pattern recognition algorithms, and machine learning approaches for predictive modeling'
            }
        ]
        
        # 执行所有测试用例
        print(f"\n" + "=" * 60)
        print("开始多用例对比测试")
        print("=" * 60)
        
        all_results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎯 测试案例 {i}: {test_case['name']}")
            print(f"   消息段: {test_case['message_segments']}")
            print(f"   完整消息: {''.join(test_case['message_segments'])}")
            print(f"   提示文本: '{test_case['prompt']}'")
            
            # 执行测试
            result = test_cyclic_embedding(
                test_case['message_segments'], 
                test_case['prompt'], 
                config, model, tokenizer
            )
            result['test_name'] = test_case['name']
            all_results.append(result)
            
            if i < len(test_cases):
                print(f"\n" + "-" * 40 + " 下一个测试 " + "-" * 40)
        
        # 汇总所有测试结果
        print(f"\n" + "=" * 60)
        print("📈 测试结果汇总")
        print("=" * 60)
        
        success_count = sum(1 for r in all_results if r.get('embed_success', False))
      
        
    except Exception as e:
        print(f"\n💥 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 水印衰减验证测试\n")
    
    # 检查命令行参数
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', '--attenuation']:
        print("开始水印衰减验证测试...")
        test_watermark_attenuation()
    else:
        try:
            do_test = input("是否进行水印衰减验证测试？(需要加载大模型，约5-10分钟) [y/N]: ").lower().strip()
            if do_test == 'y':
                test_watermark_attenuation()
            else:
                print("跳过测试")
        except EOFError:
            print("非交互模式，跳过测试")
            print("提示：使用 'python test_complex_messages_real.py --test' 进行测试")
    
    print("\n✅ 程序结束") 