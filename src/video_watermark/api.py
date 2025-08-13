from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from . import load as load_videoseal
from .adapters import generate_video_from_text as _t2v_generate
from .config_loader import get_video_config, read_model_card_args_nbits
from .types import VideoFrames, WatermarkBits
from .utils.io import ensure_even_size, save_video_mp4 as _save_video_mp4
from .utils.message import random_bits, assert_len


def generate_video_from_text(prompt: str, negative_prompt: Optional[str] = None) -> VideoFrames:
    return _t2v_generate(prompt, negative_prompt)


def _ensure_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model_from_cfg(cfg: dict):
    model = load_videoseal(cfg.get("model_card", "videoseal"))
    device = _ensure_device(cfg.get("device", "auto"))
    model = model.to(device)

    # Apply overrides if provided
    override = cfg.get("override", {}) or {}
    if override.get("scaling_w") is not None:
        model.blender.scaling_w = float(override["scaling_w"])  # type: ignore[attr-defined]
    if override.get("scaling_i") is not None:
        model.blender.scaling_i = float(override["scaling_i"])  # type: ignore[attr-defined]
    if override.get("videoseal_chunk_size") is not None:
        model.chunk_size = int(override["videoseal_chunk_size"])  # type: ignore[attr-defined]
    if override.get("videoseal_step_size") is not None:
        model.step_size = int(override["videoseal_step_size"])  # type: ignore[attr-defined]
    if override.get("attenuation") is not None:
        # JND name is handled inside model cards; at runtime we keep as-is
        pass
    return model, device


def embed_video_watermark(
    video: VideoFrames,
    message_bits: Optional[WatermarkBits] = None,
) -> tuple[VideoFrames, WatermarkBits]:
    cfg = get_video_config()
    model, device = _load_model_from_cfg(cfg)

    # message length must match model card nbits
    nbits = read_model_card_args_nbits(cfg.get("model_card", "videoseal_1.0"))
    if message_bits is None:
        message_bits = random_bits(nbits, device=device)
    else:
        assert_len(message_bits, nbits)

    frames = video.frames.to(device).float()
    # ensure range [0,1]
    frames = frames.clamp(0, 1)

    outputs = model.embed(frames, msgs=message_bits.bits, is_video=True, lowres_attenuation=bool(cfg.get("lowres_attenuation", True)))
    frames_w = outputs["imgs_w"].detach().to(video.frames.device)

    result = VideoFrames(frames=frames_w, fps=video.fps, size=video.size)
    return result, message_bits


def save_video_mp4(frames: VideoFrames, out_path: str, copy_audio_from: Optional[str] = None) -> str:
    return _save_video_mp4(frames, out_path, copy_audio_from)


def text_to_watermarked_video(
    prompt: str,
    negative_prompt: Optional[str] = None,
    message_bits: Optional[WatermarkBits] = None,
    out_dir: Optional[str] = None,
) -> dict:
    cfg = get_video_config()
    out_root = Path(out_dir or cfg.get("output_dir", "outputs")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # step 1: generate video (placeholder)
    video = generate_video_from_text(prompt, negative_prompt)

    # step 2: embed watermark
    video_w, bits = embed_video_watermark(video, message_bits)

    # step 3: save mp4 and .txt
    base = (out_root / ("video_wm.mp4")).as_posix()
    video_path = save_video_mp4(video_w, base)
    msg_path = base.replace(".mp4", ".txt")
    with open(msg_path, "w", encoding="utf-8") as f:
        f.write("".join(map(str, bits.bits.view(-1).tolist())))

    return {
        "video_path": video_path,
        "message_path": msg_path,
        "nbits": bits.length,
        "fps": video_w.fps,
        "size": video_w.size,
    }


