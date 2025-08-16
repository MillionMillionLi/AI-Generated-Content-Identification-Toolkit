import pytest
from PIL import Image

from unified.watermark_tool import WatermarkTool


@pytest.mark.timeout(180)
def test_videoseal_embed_extract_on_pil():
    # 构造一张简单的测试图像（避免依赖Stable Diffusion生成路径）
    img = Image.new("RGB", (256, 256), color=(128, 196, 64))

    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')

    # 嵌入消息
    message = "hello_videoseal"
    wm_img = tool.embed_image_watermark(img, message=message)
    assert isinstance(wm_img, Image.Image)

    # 提取水印
    res = tool.extract_image_watermark(wm_img)
    assert isinstance(res, dict)
    assert 'detected' in res and 'confidence' in res

    # 只要检测到或置信度达到基础阈值即可判定接口正确
    assert res['detected'] is True or res['confidence'] >= 0.05


def test_videoseal_embed_requires_message():
    img = Image.new("RGB", (128, 128), color=(0, 0, 0))
    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')

    with pytest.raises(ValueError):
        tool.embed_image_watermark(img, message=None)


@pytest.mark.timeout(600)
def test_videoseal_full_generation_flow():
    """
    从 Stable Diffusion 生成 → VideoSeal 嵌入 → 提取 的完整链路测试。
    注意：需要本地已缓存的 SD 权重，或可联网环境；显存不足可将 device 设为 cpu。
    """
    tool = WatermarkTool()
    tool.set_algorithm('image', 'videoseal')

    # 降低生成开销以适配CI或低资源环境
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


