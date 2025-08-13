from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

try:
    import av  # type: ignore
except Exception:
    av = None  # type: ignore
import cv2  # type: ignore
import torch

from ..types import VideoFrames


def ensure_even_size(h: int, w: int) -> Tuple[int, int]:
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    if h2 == 0:  # fallback
        h2 = 2
    if w2 == 0:
        w2 = 2
    return h2, w2


def save_video_mp4(frames: VideoFrames, out_path: str, copy_audio_from: Optional[str] = None) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    h, w = frames.size
    h2, w2 = ensure_even_size(h, w)

    # Resize if needed for even size compatibility
    vid = frames.frames
    if (h2, w2) != (h, w):
        vid = torch.nn.functional.interpolate(
            vid.permute(1, 0, 2, 3).unsqueeze(0),  # [1,3,T,H,W]
            size=(h2, w2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 0, 2, 3).contiguous()

    if av is not None:
        # Write via PyAV
        container = av.open(out_path, mode="w")
        stream = container.add_stream("libx264", rate=frames.fps)
        stream.width = w2
        stream.height = h2
        stream.pix_fmt = "yuv420p"
        for t in range(vid.shape[0]):
            img = (vid[t].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            packet = stream.encode(frame)
            if packet:
                container.mux(packet)
        packet = stream.encode(None)
        if packet:
            container.mux(packet)
        container.close()
    else:
        # Fallback to OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, frames.fps, (w2, h2))
        try:
            for t in range(vid.shape[0]):
                img = (vid[t].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

    # Optionally copy audio track (best-effort: external ffmpeg may be used by users)
    return out_path


