"""
Video watermark public API.

Usage:
  from video_watermark import load
  model = load("videoseal")

This will load the default VideoSeal model (videoseal_1.0), downloading
the checkpoint to ckpts/ if needed.
"""

from .videoseal import load as load

__all__ = ["load"]


