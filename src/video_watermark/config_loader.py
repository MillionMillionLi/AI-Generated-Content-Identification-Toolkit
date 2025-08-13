from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_RUNTIME_OVERRIDES: Dict[str, Any] = {}


def set_overrides(overrides: Dict[str, Any]) -> None:
    """Set runtime overrides that will be merged into loaded config."""
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = dict(overrides or {})
    # Clear cache so next get re-reads and merges
    _load_yaml.cache_clear()  # type: ignore[attr-defined]


def clear_overrides() -> None:
    """Clear runtime overrides."""
    set_overrides({})


@lru_cache(maxsize=4)
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Provide sensible defaults if missing
    cfg.setdefault("model_card", "videoseal_1.0")
    cfg.setdefault("device", "auto")
    cfg.setdefault("short_edge_size", -1)
    cfg.setdefault("lowres_attenuation", True)
    cfg.setdefault("output_dir", "outputs/")
    cfg.setdefault("ckpts_dir", "ckpts/")
    cfg.setdefault("override", {})

    # Placeholder T2V defaults (not part of file schema but needed by adapter)
    cfg.setdefault("default_T", 16)
    cfg.setdefault("default_fps", 24)
    cfg.setdefault("default_size", 256)
    return cfg


def get_video_config(explicit_path: Optional[str] = None) -> Dict[str, Any]:
    """Load video configuration with optional runtime overrides.

    Resolution order:
    1) explicit_path if provided
    2) env UWT_VIDEO_CONFIG
    3) project default config/video_config.yaml
    """
    if explicit_path is not None:
        cfg_path = explicit_path
    else:
        cfg_path = os.environ.get(
            "UWT_VIDEO_CONFIG",
            str((Path(__file__).resolve().parents[2] / "config" / "video_config.yaml").resolve()),
        )

    cfg = _load_yaml(cfg_path)
    cfg = _ensure_defaults(cfg)

    # Merge overrides section with runtime overrides
    merged_override = dict(cfg.get("override", {}))
    merged_override.update(_RUNTIME_OVERRIDES)
    cfg["override"] = merged_override
    return cfg


def read_model_card_args_nbits(model_card_name: str) -> int:
    """Read nbits from a videoseal model card without loading the model."""
    cards_dir = Path(__file__).resolve().parent / "videoseal" / "cards"
    card_path = cards_dir / f"{model_card_name}.yaml"
    if not card_path.exists():
        # allow passing alias 'videoseal' which maps to default
        if model_card_name == "videoseal":
            card_path = cards_dir / "videoseal_1.0.yaml"
        else:
            raise FileNotFoundError(f"Model card not found: {card_path}")
    with open(card_path, "r", encoding="utf-8") as f:
        card = yaml.safe_load(f) or {}
    args = (card or {}).get("args", {})
    nbits = int(args.get("nbits", 256))
    return nbits


