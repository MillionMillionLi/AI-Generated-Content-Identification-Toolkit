from pathlib import Path

from src.video_watermark.videoseal_watermark import VideoSealWatermark
from src.video_watermark.api import save_video_mp4


def test_videoseal_e2e_embed_extract_and_save():
    vs = VideoSealWatermark()
    message = "hello"

    # 1) embed: 生成视频并嵌入字符串消息
    video_w, bits = vs.embed("A short demo clip", message=message)

    # 2) 固定输出到项目目录 test_outputs/
    proj_root = Path(__file__).resolve().parents[1]
    out_dir = proj_root / "test_outputs" / "videoseal_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = out_dir / "videoseal_demo_wm.mp4"
    save_video_mp4(video_w, str(out_mp4))
    out_bits = out_dir / "videoseal_demo_bits.txt"
    out_bits.write_text("".join(map(str, bits.bits.view(-1).tolist())), encoding="utf-8")

    assert out_mp4.exists() and out_mp4.suffix == ".mp4"
    assert out_bits.exists()

    # 3) extract: 检测与还原消息（并保存到项目目录）
    res = vs.extract(video_w)
    assert isinstance(res.get("detected"), bool)
    assert res["detected"] is True
    # assert res.get("message") == message
    print(f"Extracted bitstring: {res['bitstring']}")
    out_msg = out_dir / "videoseal_demo_message.txt"
    out_msg.write_text(res['bitstring'], encoding="utf-8")


