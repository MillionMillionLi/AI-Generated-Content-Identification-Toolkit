import torch

from src.video_watermark.videoseal_watermark import VideoSealWatermark
from src.video_watermark.config_loader import get_video_config, read_model_card_args_nbits


def test_videoseal_watermark_embed_and_extract_smoke():
    vs = VideoSealWatermark()
    # 简短字符串，保证在 nbits 容量内
    video_w, bits = vs.embed("A demo clip", message="hello")

    assert video_w.frames.ndim == 4 and video_w.frames.shape[1] == 3
    assert bits.length == read_model_card_args_nbits(get_video_config().get("model_card", "videoseal_1.0"))

    # 提取不做强要求，只检查不报错且字段存在
    res = vs.extract(video_w)
    assert isinstance(res.get("detected"), bool)
    assert "bits" in res and res["bits"].length == bits.length
    assert isinstance(res.get("confidence"), float)


