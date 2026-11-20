"""Run one question over local images and videos."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_assets(videos, images, run_dir):
    from PIL import Image
    from agents.frame_caption import process_video
    from agents.shared.video_utils import decode_video_to_frames

    assets = []
    for index, path in enumerate(videos):
        asset_id = f"v{index}"
        work = run_dir / "assets" / asset_id
        raw = work.parent / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / (asset_id + path.suffix.lower())).symlink_to(path)
        subtitle = path.with_suffix(".srt")
        if subtitle.is_file():
            (raw / (asset_id + ".srt")).symlink_to(subtitle)
        frames = decode_video_to_frames(str(path), frames_dir=str(work / "frames"))
        captions = work / "captions"
        process_video(frames, str(captions), str(subtitle) if subtitle.is_file() else None)
        assets.append(dict(asset_id=asset_id, type="video", path=str(path),
                           db_path=str(work / "database.json"), caption_path=str(captions / "captions.json")))
    for index, path in enumerate(images):
        with Image.open(path) as image:
            image.verify()
        assets.append(dict(asset_id=f"i{index}", type="image", path=str(path)))
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", default=[], help="Local video (repeatable)")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Local image (repeatable)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--run-dir", type=Path, help="New or empty directory (default: outputs/run-*)")
    parser.add_argument("--quiet", action="store_true", help="Write the step trace without printing each step")
    args = parser.parse_args()
    if not args.question.strip() or not (args.video or args.image) or args.max_iterations < 1:
        parser.error("Provide media, a non-empty question and a positive iteration limit.")
    for path in args.video + args.image:
        if not path.expanduser().is_file():
            parser.error(f"Media file does not exist: {path}")
    from agents import config
    from agents.core_agent import CoreAgent
    from agents.logging import StepLogger
    from agents.shared.utils import extract_answer

    config.validate(video=bool(args.video))
    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        if any(run_dir.iterdir()):
            raise ValueError("--run-dir must be empty. Choose a new directory.")
    else:
        outputs = ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        run_dir = Path(mkdtemp(prefix="run-", dir=outputs))
    print(f"Run directory: {run_dir}", flush=True)
    # Exclusive creation prevents concurrent runs from sharing the same output.
    with (run_dir / "result.json").open("x", encoding="utf-8") as output:
        try:
            assets = prepare_assets([p.expanduser().resolve() for p in args.video],
                                    [p.expanduser().resolve() for p in args.image], run_dir)
            agent = CoreAgent(asset_descriptors=assets, max_iterations=args.max_iterations,
                              step_logger=StepLogger(run_dir, quiet=args.quiet))
            messages = agent.run(args.question.strip()) or []
            report = agent.last_run_report
            kind = report.get("final_response_kind", "incomplete")
            text = report.get("final_output_text") or (extract_answer(messages[-1]) if messages else None)
            if kind not in {"answer", "answer_forced", "insufficient_evidence"} or not text:
                raise RuntimeError(f"Task ended without a usable result ({kind}). Check the step trace and model settings.")
            record = {"result_kind": kind, "final_answer": text,
                      "run_report_summary": {"token_usage": report.get("token_usage")}}
            print(f"{kind}: {text}", flush=True)
        except Exception as exc:
            json.dump({"result_kind": "error", "error": str(exc)}, output, ensure_ascii=False)
            raise
        json.dump(record, output, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(str(exc))
