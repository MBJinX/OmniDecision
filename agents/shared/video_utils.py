import math
import os

import cv2

import agents.config as config


def decode_video_to_frames(video_path: str, *, frames_dir: str | None = None) -> str:
    """
    Decodes a video into JPEG frames at the frame rate specified by config.VIDEO_FPS.

    Args:
        video_path: The absolute path to the video file.
        frames_dir: Optional output directory. When omitted, falls back to
            config.VIDEO_DATABASE_FOLDER/<video_name>/frames for legacy callers.

    Returns:
        The absolute path to the directory containing the extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        Exception: If frame extraction fails.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if frames_dir is None:
        frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_name, 'frames')
    else:
        frames_dir = os.path.abspath(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file '{video_path}'.")

    saved_count = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot determine the video frame rate.")
        target_fps = config.VIDEO_FPS
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            # Sample on the output clock so fractional source rates do not drift.
            while saved_count * fps < frame_count * target_fps:
                name = os.path.join(frames_dir, f"frame_n{saved_count:06d}.jpg")
                if not cv2.imwrite(name, frame):
                    raise OSError("Failed to write sampled frame.")
                saved_count += 1
    finally:
        cap.release()
    if saved_count == 0:
        raise ValueError("The input contains no decodable frames.")
    return os.path.abspath(frames_dir)
