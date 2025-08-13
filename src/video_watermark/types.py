from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class VideoFrames:
    frames: torch.Tensor  # [T, 3, H, W], float32 in [0,1]
    fps: int
    size: Tuple[int, int]  # (H, W)

    def to(self, device: torch.device | str) -> "VideoFrames":
        return VideoFrames(frames=self.frames.to(device), fps=self.fps, size=self.size)


@dataclass
class WatermarkBits:
    bits: torch.Tensor  # [1, K], dtype long or bool

    @property
    def length(self) -> int:
        return int(self.bits.shape[-1])


