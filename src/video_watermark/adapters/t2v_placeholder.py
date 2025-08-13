from __future__ import annotations

from typing import Optional

import torch

from ..config_loader import get_video_config
from ..types import VideoFrames


def generate_video_from_text(prompt: str, negative_prompt: Optional[str] = None) -> VideoFrames:
    """Placeholder T2V: return random frames tensor with shape [T,3,H,W] in [0,1]."""
    cfg = get_video_config()
    T = int(cfg.get("default_T", 16))
    fps = int(cfg.get("default_fps", 24))
    size = int(cfg.get("default_size", 256))
    frames = torch.rand(T, 3, size, size)
    return VideoFrames(frames=frames, fps=fps, size=(size, size))


