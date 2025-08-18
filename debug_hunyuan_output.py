#!/usr/bin/env python3
"""
HunyuanVideo输出调试脚本
专门用于诊断黑屏视频生成问题
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.video_watermark.hunyuan_video_generator import create_hunyuan_generator

def debug_hunyuan_output():
    """调试HunyuanVideo的原始输出"""
    
    # 设置详细日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🔍 HunyuanVideo输出调试分析")
    print("=" * 50)
    
    # 创建生成器
    generator = create_hunyuan_generator()
    
    # 更保守的测试参数（避免内存问题）
    test_params = {
        'prompt': '一朵红色的花',
        'num_frames': 9,   # 更少的帧数
        'height': 256,     # 更小的分辨率
        'width': 256,
        'num_inference_steps': 5,  # 更少的推理步数
        'seed': 42
    }
    
    print(f"测试参数: {test_params}")
    
    try:
        # 加载管道
        generator._load_pipeline()
        
        print("\n📋 管道信息:")
        print(f"  Pipeline类型: {type(generator.pipeline)}")
        print(f"  设备: {generator.device}")
        print(f"  数据类型: {generator.pipeline.dtype if hasattr(generator.pipeline, 'dtype') else 'unknown'}")
        
        # 手动调用管道，跟踪每一步
        print(f"\n🎬 开始生成视频...")
        
        with torch.no_grad():
            # 设置随机种子确保可重复性
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            result = generator.pipeline(
                prompt=test_params['prompt'],
                num_frames=test_params['num_frames'],
                height=test_params['height'],
                width=test_params['width'],
                num_inference_steps=test_params['num_inference_steps'],
                guidance_scale=6.0,
                generator=torch.Generator(device=generator.device).manual_seed(42)
            )
            
            print(f"\n📤 管道返回结果类型: {type(result)}")
            
            # 详细分析result结构
            if hasattr(result, '__dict__'):
                print(f"Result属性: {list(result.__dict__.keys())}")
                for attr_name in result.__dict__.keys():
                    attr_value = getattr(result, attr_name)
                    print(f"  {attr_name}: {type(attr_value)}")
                    
                    if attr_name == 'frames' and attr_value is not None:
                        print(f"    frames详情:")
                        print(f"      类型: {type(attr_value)}")
                        print(f"      长度: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}")
                        
                        if hasattr(attr_value, '__len__') and len(attr_value) > 0:
                            first_batch = attr_value[0]
                            print(f"      first_batch类型: {type(first_batch)}")
                            print(f"      first_batch长度: {len(first_batch) if hasattr(first_batch, '__len__') else 'N/A'}")
                            
                            if hasattr(first_batch, '__len__') and len(first_batch) > 0:
                                first_frame = first_batch[0]
                                print(f"      first_frame类型: {type(first_frame)}")
                                
                                if hasattr(first_frame, 'size'):
                                    print(f"      first_frame大小: {first_frame.size}")
                                    
                                    # 转换为numpy分析像素值
                                    frame_array = np.array(first_frame)
                                    print(f"      numpy形状: {frame_array.shape}")
                                    print(f"      数据类型: {frame_array.dtype}")
                                    print(f"      值域: [{frame_array.min():.3f}, {frame_array.max():.3f}]")
                                    print(f"      平均值: {frame_array.mean():.3f}")
                                    print(f"      标准差: {frame_array.std():.3f}")
                                    
                                    # 检查是否包含NaN或inf
                                    has_nan = np.isnan(frame_array).any()
                                    has_inf = np.isinf(frame_array).any()
                                    print(f"      包含NaN: {has_nan}")
                                    print(f"      包含Inf: {has_inf}")
                                    
                                    # 分析像素分布
                                    unique_values = np.unique(frame_array)
                                    print(f"      唯一值数量: {len(unique_values)}")
                                    if len(unique_values) <= 10:
                                        print(f"      唯一值: {unique_values}")
                                    
                                    # 检查每个通道
                                    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                                        for i, color in enumerate(['R', 'G', 'B']):
                                            channel = frame_array[:, :, i]
                                            print(f"      {color}通道: [{channel.min():.3f}, {channel.max():.3f}], 均值={channel.mean():.3f}")
            
            # 尝试获取视频帧
            if hasattr(result, 'frames') and result.frames is not None:
                video_frames = result.frames[0]
            elif hasattr(result, 'videos') and result.videos is not None:
                video_frames = result.videos[0]
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                video_frames = result[0]
            else:
                video_frames = result
                
            print(f"\n🎞️ 提取的video_frames:")
            print(f"  类型: {type(video_frames)}")
            
            if isinstance(video_frames, list) and len(video_frames) > 0:
                print(f"  列表长度: {len(video_frames)}")
                print(f"  第一帧类型: {type(video_frames[0])}")
                
                # 检查是否所有帧都是黑色的
                all_black = True
                for i, frame in enumerate(video_frames[:3]):  # 检查前3帧
                    if hasattr(frame, 'convert'):
                        # PIL Image
                        img_array = np.array(frame.convert('RGB'))
                        max_val = img_array.max()
                        mean_val = img_array.mean()
                        print(f"  帧{i}: 最大值={max_val}, 平均值={mean_val:.3f}")
                        if max_val > 0:
                            all_black = False
                
                print(f"  是否全黑: {all_black}")
            
            # 保存一个测试帧查看
            if isinstance(video_frames, list) and len(video_frames) > 0:
                test_frame = video_frames[0]
                if hasattr(test_frame, 'save'):
                    test_frame.save("debug_first_frame.png")
                    print(f"\n💾 第一帧已保存为 debug_first_frame.png")
                    
                    # 检查保存的文件
                    import os
                    file_size = os.path.getsize("debug_first_frame.png") / 1024
                    print(f"  文件大小: {file_size:.1f} KB")
    
    except Exception as e:
        print(f"\n❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hunyuan_output()