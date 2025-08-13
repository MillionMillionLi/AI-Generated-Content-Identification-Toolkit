"""
VideoSeal 统一封装类：提供最简单的 embed / extract 接口

用法示例：
    from src.video_watermark.types import VideoFrames
    from src.video_watermark.videoseal_watermark import VideoSealWatermark

    vs = VideoSealWatermark()
    video = ...  # VideoFrames[T,3,H,W]
    video_w, bits = vs.embed(video)           # 嵌入
    result = vs.extract(video_w)              # 提取
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import load as load_videoseal
from .config_loader import get_video_config, read_model_card_args_nbits
from .types import VideoFrames, WatermarkBits
from .utils.message import random_bits, encode_string_to_bits, decode_bits_to_string, to_bitstring
from .adapters import generate_video_from_text


def _ensure_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class VideoSealWatermark:
    """最小封装：加载模型卡 → 提供 embed / extract 接口。"""

    def __init__(self, cfg: Optional[dict] = None) -> None:
        self.cfg = cfg or get_video_config()
        self.device = _ensure_device(self.cfg.get("device", "auto"))
        self.model = load_videoseal(self.cfg.get("model_card", "videoseal"))
        self.model = self.model.to(self.device)

        # 应用可选覆盖项
        override = self.cfg.get("override", {}) or {}
        if override.get("scaling_w") is not None:
            self.model.blender.scaling_w = float(override["scaling_w"])  # type: ignore[attr-defined]
        if override.get("scaling_i") is not None:
            self.model.blender.scaling_i = float(override["scaling_i"])  # type: ignore[attr-defined]
        if override.get("videoseal_chunk_size") is not None:
            self.model.chunk_size = int(override["videoseal_chunk_size"])  # type: ignore[attr-defined]
        if override.get("videoseal_step_size") is not None:
            self.model.step_size = int(override["videoseal_step_size"])  # type: ignore[attr-defined]

        # 记录模型卡 nbits
        self.nbits = read_model_card_args_nbits(self.cfg.get("model_card", "videoseal_1.0"))

    @torch.no_grad()
    def embed(self, prompt: str, message: Optional[str] = None, negative_prompt: Optional[str] = None) -> Tuple[VideoFrames, WatermarkBits]:
        """根据 prompt 生成视频并嵌入字符串消息对应的比特水印。

        Args:
            prompt: 文生视频提示词
            message: 要嵌入的消息（字符串）。若为空则随机生成比特串
            negative_prompt: 负向提示词（当前占位生成器未使用）
        Returns:
            (VideoFrames, WatermarkBits): 带水印的视频与消息位
        """
        # 1) 生成占位视频
        video = generate_video_from_text(prompt, negative_prompt)

        # 2) 生成/转换消息比特（可逆编码：长度头 + UTF-8 内容）
        if message is None:
            message_bits = random_bits(self.nbits, self.device)
        else:
            message_bits = encode_string_to_bits(message, self.nbits, device=self.device)

        # 3) 嵌入
        frames = video.frames.to(self.device).float().clamp(0, 1)
        outputs = self.model.embed(
            frames,
            msgs=message_bits.bits.to(self.device),
            is_video=True,
            lowres_attenuation=bool(self.cfg.get("lowres_attenuation", True)),
        )
        frames_w = outputs["imgs_w"].detach().to(video.frames.device)
        return VideoFrames(frames=frames_w, fps=video.fps, size=video.size), message_bits

    @torch.no_grad()
    def extract(self, video: VideoFrames, aggregation: str = "avg") -> dict:
        """从视频中提取水印消息。

        返回：
            {
              'detected': bool,
              'bits': WatermarkBits,
              'confidence': float
            }
        """
        frames = video.frames.to(self.device).float().clamp(0, 1)
        # 计算帧级预测
        outputs = self.model.detect(frames, is_video=True)
        preds = outputs["preds"]  # [F, 1+K, ...] 或 [F, 1+K]

        # 估计检测置信度：使用检测通道(第0列) > 0 的比例
        det = preds[:, 0]
        while det.dim() > 1:
            det = det.mean(dim=-1)
        confidence = float((det > 0).float().mean().item())
        detected = confidence >= 0.5

        # 聚合消息位并尝试解码为字符串（多策略聚合以提升还原成功率）
        agg_methods = []
        if aggregation:
            agg_methods.append(aggregation)
        for m in ["squared_avg", "l1norm_avg", "l2norm_avg", None]:
            if m not in agg_methods:
                agg_methods.append(m)

        decoded_message: Optional[str] = None
        final_bits = None
        agg_used: Optional[str] = None
        for m in agg_methods:
            msg = self.model.extract_message(frames, aggregation=m)
            bits = msg
            if bits.dtype != torch.long:
                bits = (bits > 0).long()
            bits = bits.to(video.frames.device)
            try:
                decoded_message = decode_bits_to_string(WatermarkBits(bits=bits))
                final_bits = bits
                agg_used = m if isinstance(m, str) else "none"
                break
            except Exception:
                continue

        if final_bits is None:
            msg = self.model.extract_message(frames, aggregation=aggregation)
            final_bits = (msg > 0).long().to(video.frames.device)
            agg_used = aggregation if isinstance(aggregation, str) else "avg"

        result = {
            "detected": bool(detected),
            "bits": WatermarkBits(bits=final_bits),
            "confidence": confidence,
            "aggregation_used": agg_used,
            "bitstring": to_bitstring(WatermarkBits(bits=final_bits)),
        }
        if decoded_message is not None:
            result["message"] = decoded_message
        return result


