#!/usr/bin/env python3
"""
精简版：HunyuanVideo + VideoSeal 演示/回归测试
- 依赖统一接口 `src.video_watermark.VideoWatermark`
- 加载本地 HunyuanVideo 快照（由底层生成器负责）
- 两项用例：
  1) 纯文生视频（阳光洒在海面上）
  2) 文生视频 + 水印嵌入 + 提取验证
- 保守的默认参数，可通过配置覆盖
"""

import os
import sys
import time
import logging
from pathlib import Path
import yaml

# 项目根目录入路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.video_watermark import VideoWatermark
    from src.video_watermark.utils import PerformanceTimer, FileUtils, VideoIOUtils
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请在项目根目录运行该脚本，或检查PYTHONPATH")
    sys.exit(1)


def load_config():
    cfg_path = project_root / "config" / "video_config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging():
    FileUtils.ensure_dir("tests/test_results")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tests/test_results/demo.log')
        ]
    )


def check_models_and_system(config):
    print("=" * 60)
    print("🔧 测试1：模型与系统检查")
    print("=" * 60)

    cache_dir = (config or {}).get('system', {}).get(
        'cache_dir',
        "/fs-computility/wangxuhong/limeilin/.cache/huggingface/hub"
    )

    try:
        with PerformanceTimer("初始化VideoWatermark"):
            wm = VideoWatermark(cache_dir=cache_dir, config=config)

        sys_info = wm.get_system_info()
        print("\n📊 系统信息:")
        for k, v in sys_info.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")

        model_manager = wm._ensure_model_manager()
        with PerformanceTimer("确保Hunyuan本地模型可用"):
            local_path = model_manager.ensure_hunyuan_model(allow_download=False)
        print(f"✅ 本地Hunyuan模型: {local_path}")
        return True, wm

    except Exception as e:
        print(f"❌ 模型/系统检查失败: {e}")
        return False, None


def test_text_to_video(wm: VideoWatermark, config):
    print("\n" + "=" * 60)
    print("🎬 测试2：Hunyuan 纯文生视频")
    print("=" * 60)

    cfg_params = ((config or {}).get('hunyuan_video') or {}).get('test_params') or {}
    params = {
        'num_frames': int(cfg_params.get('num_frames', 75)),
        'height': int(cfg_params.get('height', 320)),
        'width': int(cfg_params.get('width', 512)),
        'num_inference_steps': int(cfg_params.get('num_inference_steps', 30)),
    }
    fps = int(cfg_params.get('fps', 15))

    prompt = "阳光洒在海面上"
    print(f"提示词: {prompt}")

    try:
        gen = wm._ensure_video_generator()
        with PerformanceTimer("视频生成"):
            video_tensor = gen.generate_video_tensor(prompt=prompt, seed=42, **params)

        # 基础质量检查：非全黑/非全零
        arr = video_tensor.detach().cpu().numpy()
        min_v, max_v, mean_v = float(arr.min()), float(arr.max()), float(arr.mean())
        print(f"📈 像素统计: min={min_v:.4f} max={max_v:.4f} mean={mean_v:.4f}")
        if max_v <= 0.0 or mean_v <= 0.001:
            raise RuntimeError("生成结果疑似全黑/无效")

        out_path = "tests/test_results/test_generation_1.mp4"
        FileUtils.ensure_dir(Path(out_path).parent)
        VideoIOUtils.save_video_tensor(video_tensor, out_path, fps=fps)
        size_mb = FileUtils.get_file_size_mb(out_path)

        print(f"✅ 生成完成: {out_path}")
        print(f"📹 视频信息: {tuple(video_tensor.shape)}, 文件大小: {size_mb:.1f} MB")
        return True, out_path

    except Exception as e:
        print(f"❌ 文生视频失败: {e}")
        return False, ""


def test_text_to_video_with_watermark(wm: VideoWatermark, config):
    print("\n" + "=" * 60)
    print("🔐 测试3：文生视频 + 水印嵌入 + 提取")
    print("=" * 60)

    demo_params = ((config or {}).get('demo') or {}).get('demo_params') or {}
    params = {
        'num_frames': int(demo_params.get('num_frames', 75)),
        'height': int(demo_params.get('height', 320)),
        'width': int(demo_params.get('width', 512)),
        'num_inference_steps': int(demo_params.get('num_inference_steps', 30)),
    }

    prompt = "阳光洒在海面上"
    message = "demo_msg"
    print(f"提示词: {prompt}")
    print(f"水印: {message}")

    try:
        with PerformanceTimer("完整流程"):
            out_path = wm.generate_video_with_watermark(
                prompt=prompt,
                message=message,
                seed=100,
                **params
            )
        size_mb = FileUtils.get_file_size_mb(out_path)
        print(f"✅ 生成完成: {out_path} ({size_mb:.1f} MB)")

        with PerformanceTimer("水印提取"):
            result = wm.extract_watermark(out_path, max_frames=50)
        print(f"🔎 提取: detected={result['detected']} message='{result.get('message','')}' confidence={result['confidence']:.3f}")

        ok = bool(result['detected'])
        return ok, out_path

    except Exception as e:
        print(f"❌ 完整流程失败: {e}")
        return False, ""


def main():
    print("🎬 HunyuanVideo + VideoSeal 简化测试")
    print("=" * 60)

    setup_logging()
    config = load_config()
    if not config:
        print("⚠️ 未找到配置文件，使用默认保守参数")

    start = time.time()

    ok, wm = check_models_and_system(config)
    if not ok:
        return

    ok_gen, gen_path = test_text_to_video(wm, config)
    ok_wm, wm_path = test_text_to_video_with_watermark(wm, config)

    print("\n🧹 清理内存...")
    wm.clear_cache()

    elapsed = time.time() - start
    print("\n📊 汇总:")
    print(f"  纯文生视频: {'✅' if ok_gen else '❌'} {gen_path}")
    print(f"  文生+水印: {'✅' if ok_wm else '❌'} {wm_path}")
    print(f"⏱️ 总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()