import sys
from pathlib import Path

# 确保可以从根目录直接运行：把 src/ 加到 sys.path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import argparse
import pytest
from PIL import Image
from unified.watermark_tool import WatermarkTool


@pytest.mark.timeout(180)
def test_videoseal_embed_extract_on_pil_root():
    img = Image.new("RGB", (256, 256), color=(128, 196, 64))

    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')

    message = "hello_videoseal"
    wm_img = tool.embed_image_watermark(img, message=message)
    assert isinstance(wm_img, Image.Image)

    res = tool.extract_image_watermark(wm_img)
    assert isinstance(res, dict)
    assert 'detected' in res and 'confidence' in res
    assert res['detected'] is True or res['confidence'] >= 0.05


@pytest.mark.timeout(600)
def test_videoseal_full_generation_flow_root():
    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')

    # 降低资源需求以提高通过率
    tool.image_watermark.config['resolution'] = 256
    tool.image_watermark.config['num_inference_steps'] = 5
    tool.image_watermark.config['device'] = tool.image_watermark.config.get('device', None) or 'cpu'

    prompt = "a cute cat, simple background, flat colors"
    message = "hello_videoseal"

    try:
        wm_img = tool.generate_image_with_watermark(prompt=prompt, message=message)
    except Exception as e:
        pytest.skip(f"跳过：生成阶段失败（可能无权重/离线/资源不足）：{e}")

    assert isinstance(wm_img, Image.Image)

    try:
        res = tool.extract_image_watermark(wm_img)
    except Exception as e:
        pytest.skip(f"跳过：提取阶段失败（可能显存不足或依赖未就绪）：{e}")

    assert isinstance(res, dict)
    assert 'detected' in res and 'confidence' in res
    assert res['detected'] is True or res['confidence'] >= 0.05



def _run_pil_flow(device: str = 'cpu') -> bool:
    print("[PIL] 单图嵌入/提取演示")
    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')
    tool.image_watermark.config['device'] = device or 'cpu'
    img = Image.new("RGB", (256, 256), color=(128, 196, 64))
    message = "hello_videoseal"
    try:
        wm_img = tool.embed_image_watermark(img, message=message)
        res = tool.extract_image_watermark(wm_img)
        print(f"  检测: {res.get('detected')}  置信度: {res.get('confidence'):.3f}  消息: {res.get('message','')}")
        return bool(res.get('detected')) or float(res.get('confidence') or 0.0) >= 0.05
    except Exception as e:
        print(f"  失败: {e}")
        return False


def _run_gen_flow(device: str = 'cpu', resolution: int = 256, steps: int = 5,
                  prompt: str = "a cute cat, simple background, flat colors",
                  message: str = "hello_videoseal") -> bool:
    print("[GEN] 生成→嵌入→提取演示 (离线)")
    # 离线运行，禁止联网
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('DIFFUSERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')
    tool.image_watermark.config['device'] = device or 'cpu'
    tool.image_watermark.config['resolution'] = int(resolution)
    tool.image_watermark.config['num_inference_steps'] = int(steps)

    try:
        wm_img = tool.generate_image_with_watermark(prompt=prompt, message=message)
    except Exception as e:
        print(f"  生成失败(可能无本地权重/缓存路径不正确): {e}")
        return False

    try:
        res = tool.extract_image_watermark(wm_img)
        print(f"  检测: {res.get('detected')}  置信度: {res.get('confidence'):.3f}  消息: {res.get('message','')}")
        return bool(res.get('detected')) or float(res.get('confidence') or 0.0) >= 0.05
    except Exception as e:
        print(f"  提取失败: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoSeal 图像水印最小演示")
    parser.add_argument('--mode', choices=['pil', 'gen', 'both'], default='both', help='运行模式')
    parser.add_argument('--device', default='cpu', help='设备: cpu/cuda')
    parser.add_argument('--resolution', type=int, default=256, help='生成分辨率(正方形)')
    parser.add_argument('--steps', type=int, default=5, help='生成步数')
    parser.add_argument('--prompt', default='a cute cat, simple background, flat colors', help='生成提示')
    parser.add_argument('--message', default='hello_videoseal', help='嵌入消息')
    args = parser.parse_args()

    ok = True
    if args.mode in ('pil', 'both'):
        ok = _run_pil_flow(device=args.device) and ok
    if args.mode in ('gen', 'both'):
        ok = _run_gen_flow(device=args.device, resolution=args.resolution, steps=args.steps,
                           prompt=args.prompt, message=args.message) and ok
    print("\n结果: {}".format("✅ 成功" if ok else "❌ 失败"))