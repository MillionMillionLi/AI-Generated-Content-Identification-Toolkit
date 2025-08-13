# VideoSeal 集成计划（先无文生视频模型）

目标：先把公共接口、隐式配置加载与测试跑通；视频生成用占位（随机帧），后续接入真实 T2V 模型仅替换生成器实现。

## 模块与文件
- `src/video_watermark/config_loader.py`：隐式加载 `config/video_config.yaml`；支持 `UWT_VIDEO_CONFIG` 覆盖路径；全局缓存；`set_overrides`/`clear_overrides` 可选。
- `src/video_watermark/types.py`：
  - `VideoFrames(frames: torch.Tensor[T,3,H,W], fps: int, size: (H,W))`
  - `WatermarkBits(bits: torch.Tensor[1,K])`
- `src/video_watermark/api.py`：公共 API（内部 `get_video_config()`，函数签名不带配置）。
  - `generate_video_from_text(prompt, negative_prompt=None) -> VideoFrames`（占位：随机帧）
  - `embed_video_watermark(video: VideoFrames, message_bits: WatermarkBits|None=None) -> (VideoFrames, WatermarkBits)`
  - `save_video_mp4(frames: VideoFrames, out_path: str, copy_audio_from: str|None=None) -> str`
  - `text_to_watermarked_video(prompt, negative_prompt=None, message_bits=None) -> dict`
- `tests/test_video_watermark_api.py`：单测 + 最小集成测试。

## 目录结构（保留 videoseal 在 video_watermark 下）
```
src/
  video_watermark/
    __init__.py               # 对外入口（已暴露 load）
    config_loader.py          # 隐式读取 config/video_config.yaml（本模块内）
    types.py                  # VideoFrames / WatermarkBits
    api.py                    # 高层 API（不带配置参数）
    videoseal/                # 第三方 VideoSeal 包代码（位置不变）
      cards/
      configs/
      models/
      modules/
      data/
      utils/
      evals/
    adapters/                 # 文生视频适配层（可插拔）
      __init__.py
      t2v_placeholder.py      # 无 T2V 时的占位随机帧生成器
      # t2v_modelscope.py     # 预留：具体 T2V 模型适配
    utils/
      __init__.py
      io.py                   # ffmpeg 写 mp4、尺寸偶数对齐、视频读写
      message.py              # 比特串生成/校验/转换（Tensor/str 互转）

examples/
  video_demo.py               # 生成占位视频 → 嵌入 → 导出 mp4/.txt
  text2vid_wm_demo.py         # 文本 → 占位视频 → 水印 → 导出（端到端）

tests/
  test_video_wm_api.py        # API 层：形状、bits 长度、落盘校验
  test_videoseal_wrapper.py   # 封装与覆盖项：nbits 校验、override 生效
```

## 代码实现清单（函数/类）
- `video_watermark/types.py`
  - `VideoFrames(frames: Tensor[T,3,H,W], fps: int, size: tuple[int,int])`
  - `WatermarkBits(bits: Tensor[1,K])`

- `video_watermark/config_loader.py`
  - `get_video_config() -> dict`：默认读取 `config/video_config.yaml` 或 `UWT_VIDEO_CONFIG`
  - `set_overrides(d: dict)` / `clear_overrides()`：运行时临时覆盖 `override` 字段（可选）

- `video_watermark/adapters/t2v_placeholder.py`
  - `generate_video_from_text(prompt: str, negative_prompt: str | None) -> VideoFrames`
    - 读取 cfg（如 T、fps、H/W 可从 cfg 或固定常量，例如 T=16,fps=24,H=W=256）
    - 返回 [T,3,H,W] 随机帧、值域 [0,1]

- `video_watermark/utils/io.py`
  - `ensure_even_size(h: int, w: int) -> tuple[int,int]`
  - `save_video_mp4(frames: VideoFrames, out_path: str, copy_audio_from: str | None) -> str`

- `video_watermark/utils/message.py`
  - `random_bits(nbits: int, device: torch.device) -> WatermarkBits`
  - `to_bitstring(bits: WatermarkBits) -> str`
  - `from_bitstring(s: str) -> WatermarkBits`
  - `assert_len(bits: WatermarkBits, nbits: int)`

- `video_watermark/api.py`
  - `generate_video_from_text(prompt: str, negative_prompt: str | None = None) -> VideoFrames`
    - 内部调用 adapters 占位生成器
  - `embed_video_watermark(video: VideoFrames, message_bits: WatermarkBits | None = None) -> tuple[VideoFrames, WatermarkBits]`
    - `from video_watermark import load` 加载模型卡
    - 按 cfg.override 设置 scaling_w/chunk_size/step_size/attenuation（如支持）
    - 校验/生成消息长度 = nbits；调用 `model.embed(imgs, is_video=True, lowres_attenuation=cfg["lowres_attenuation"])`
  - `save_video_mp4(frames: VideoFrames, out_path: str, copy_audio_from: str | None = None) -> str`
    - 调用 utils.io 保存；保证偶数尺寸
  - `text_to_watermarked_video(prompt: str, negative_prompt: str | None = None, message_bits: WatermarkBits | None = None) -> dict`
    - 串联 生成 → 嵌入 → 导出，返回 `video_path/message_path/nbits/fps/size`

## 测试覆盖点（详细）
- API：
  - 生成占位视频的形状/值域/fps
  - 嵌入后帧形状不变，返回的消息长度=nbits
  - 端到端导出 mp4/.txt 文件存在
- 封装与覆盖项：
  - 配置 `override.scaling_w` 更改是否生效（可读取/断言模型参数或运行日志）
  - `set_overrides`/`clear_overrides` 对 cfg 的影响
- I/O：
  - 奇数尺寸输入时导出成功且自动对齐为偶数

## 接口要点
- 模型加载：`from video_watermark import load` → `load(cfg["model_card"])`。
- 覆盖：用 `cfg["override"]`（如 `scaling_w`、`videoseal_chunk_size`、`videoseal_step_size`、`attenuation`）。
- 尺寸：写 mp4 前保证宽高为偶数；必要时裁剪/缩放对齐。
- 消息：长度必须等于模型卡 `args.nbits`（默认 256）；不匹配则报错（先不做自动填充/截断）。

## 配置加载（草案）
- 默认：`config/video_config.yaml`；可用环境变量 `UWT_VIDEO_CONFIG` 指定其它路径。
- 伪代码：
```
@lru_cache(maxsize=1)
_load_yaml(path) -> dict
get_video_config(): 读 env 或默认路径 -> 合并 runtime overrides -> 返回
```

## 测试计划
- `test_generate_video_from_text_placeholder`：返回 `VideoFrames`；形状/值域/fps 合法。
- `test_embed_video_watermark_shapes_and_bits`：随机帧 → 嵌入；形状不变；bits 长度=nbits。
- `test_text_to_watermarked_video_outputs`：端到端返回包含 `video_path`/`message_path` 且文件存在。
- 覆盖用例：`set_overrides({"scaling_w":0.25})` 生效；`clear_overrides()` 恢复。

## 使用示例（占位 T2V）
```
from video_watermark.api import (
  generate_video_from_text, embed_video_watermark, save_video_mp4, text_to_watermarked_video,
)
video = generate_video_from_text("A running dog in the park")
video_w, bits = embed_video_watermark(video)
out = save_video_mp4(video_w, "outputs/demo_wm.mp4")
with open(out.replace(".mp4", ".txt"), "w") as f:
    f.write("".join(map(str, bits.bits[0].tolist())))
res = text_to_watermarked_video("Sunset over the ocean")
```

## 里程碑
- [ ] 新增 `config_loader.py`
- [ ] 新增 `types.py`
- [ ] 新增 `api.py`（占位 T2V + 嵌入 + 导出）
- [ ] 新增 `tests/test_video_watermark_api.py`
- [ ] 文档：`CLAUDE.md` 增加“视频水印 API 使用示例（占位 T2V）”
- [ ] 跑测与修复
- [ ] 后续：接入真实 T2V，补充评测
