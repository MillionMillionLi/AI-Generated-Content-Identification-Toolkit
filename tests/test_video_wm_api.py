import os
from pathlib import Path

import torch

from src.video_watermark.api import (
    generate_video_from_text,
    embed_video_watermark,
    save_video_mp4,
    text_to_watermarked_video,
)
from src.video_watermark.config_loader import get_video_config, set_overrides, clear_overrides, read_model_card_args_nbits
from src.video_watermark.types import VideoFrames
from src.video_watermark.utils.message import random_bits


def test_generate_video_from_text_placeholder():
    video = generate_video_from_text("A dog running")
    assert isinstance(video, VideoFrames)
    assert video.frames.ndim == 4 and video.frames.shape[1] == 3
    assert video.frames.min() >= 0 and video.frames.max() <= 1
    assert video.fps > 0 and isinstance(video.size, tuple)


def test_embed_video_watermark_shapes_and_bits(tmp_path: Path):
    cfg = get_video_config()
    T = 8
    H, W = 123, 257  # intentionally odd sizes
    frames = torch.rand(T, 3, H, W)
    video = VideoFrames(frames=frames, fps=cfg.get("default_fps", 24), size=(H, W))

    nbits = read_model_card_args_nbits(cfg.get("model_card", "videoseal_1.0"))
    bits = random_bits(nbits)

    video_w, bits_out = embed_video_watermark(video, bits)
    assert video_w.frames.shape == video.frames.shape
    assert bits_out.length == nbits


def test_text_to_watermarked_video_outputs(tmp_path: Path):
    out = text_to_watermarked_video("Sunset over the ocean", out_dir=str(tmp_path))
    assert os.path.exists(out["video_path"]) and out["video_path"].endswith(".mp4")
    assert os.path.exists(out["message_path"]) and out["message_path"].endswith(".txt")
    assert out["nbits"] > 0


def test_overrides_affect_model_runtime():
    # Only check override plumbing does not crash; verifying effect precisely requires peeking internals
    set_overrides({"scaling_w": 0.25, "videoseal_chunk_size": 8})
    video = generate_video_from_text("A cat")
    _ = embed_video_watermark(video)
    clear_overrides()


