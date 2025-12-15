import os
import re
import json
import argparse
import shutil
import tempfile
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# =========================
# 1. Dataset & metrics
# =========================

def load_movienet_json(path: str) -> List[Dict[str, Any]]:
    """
    Load MovieNet_Gemini_500_dataset.json

    Expected format:
    {
        "fps": 30,  # (optional)
        "segments": [
            {
                "movie_id": "ttxxxxxx",
                "description": "...",
                "start_frame_id": int,
                "end_frame_id": int,
                ...
            },
            ...
        ]
    }

    Return a flat list of dicts:
        {
          "movie_id": str,
          "description": str,
          "start_shot": int,
          "end_shot": int
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    segments = raw["segments"]
    data: List[Dict[str, Any]] = []
    for seg in segments:
        data.append(
            {
                "movie_id": seg["movie_id"],
                "description": seg["description"],
                "start_shot": int(seg["start_shot"]),
                "end_shot": int(seg["end_shot"]),
            }
        )
    return data


def temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """IoU between two temporal segments in *seconds*."""
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return inter / union


def compute_metrics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mIoU and Recall@0.3/0.5/0.7 from per-sample results (seconds)."""
    ious = []
    for s in samples:
        gt_s, gt_e = s["gt_start"], s["gt_end"]
        ps, pe = s.get("pred_start_sec", 0.0), s.get("pred_end_sec", 0.0)
        iou = temporal_iou(gt_s, gt_e, ps, pe)
        s["iou"] = iou  # store back for logging
        ious.append(iou)

    if not ious:
        return {
            "mIoU": 0.0,
            "Recall@0.3": 0.0,
            "Recall@0.5": 0.0,
            "Recall@0.7": 0.0,
        }

    m_iou = float(sum(ious) / len(ious))

    def recall_at(th: float) -> float:
        return float(sum(i >= th for i in ious) / len(ious))

    return {
        "mIoU": m_iou,
        "Recall@0.3": recall_at(0.3),
        "Recall@0.5": recall_at(0.5),
        "Recall@0.7": recall_at(0.7),
    }


# =========================
# 2. Frame-number overlay
# =========================

def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Try to load a TTF font; fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def annotate_frame_with_index(
    frame_bgr: np.ndarray,
    text: str,
    position: str = "bottom_right",
    font_size: int = 32,
    color: str = "red",
) -> np.ndarray:
    """
    Draw frame index text on the frame using PIL at given position.
    Position: top_left, top_right, bottom_left, bottom_right, center.
    """
    frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame)
    font = load_font(font_size)

    width, height = frame.size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    margin = 0
    if position == "top_left":
        x, y = margin, margin
    elif position == "top_right":
        x, y = width - text_width - margin, margin
    elif position == "bottom_left":
        x, y = margin, height - text_height - margin
    elif position == "bottom_right":
        x, y = width - text_width - margin, height - text_height - margin
    elif position == "center":
        x, y = (width - text_width) // 2, (height - text_height) // 2
    else:
        raise ValueError(f"Invalid position: {position}")

    if position in ["bottom_left", "bottom_right"]:
        y -= text_height / 3

    draw.text((x, y), text, font=font, fill=color)
    frame_bgr_out = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame_bgr_out


def annotate_and_save_video(
    src_path: str,
    dst_path: str,
    position: str = "bottom_right",
    font_size: int = 40,
    color: str = "red",
) -> Tuple[float, int]:
    """
    Read original video from src_path, overlay frame index (bottom-right),
    and save annotated video to dst_path.

    Returns:
        (orig_fps, total_frames)
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {src_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0  # fallback

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = annotate_frame_with_index(
            frame, str(frame_idx), position=position, font_size=font_size, color=color
        )
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return fps, total_frames


# =========================
# 3. Prompt & parsing  （維持原樣）
# =========================

def build_prompt(query: str, total_frames: float) -> str:
    """
    English prompt.

    - Mention bottom-right number is the frame index.
    - Ask model to answer 'From frame X to frame Y'.
    - Do NOT mention duration / fps.
    """
    prompt = (
        "The red numbers on each frame represent the frame number.\n"
        f"The total number of frame is {total_frames}\n"
        f"In this video, during which frames can we see {query}? "
        "Answer in the format: 'From Frame x to Frame y'"
    )
    return prompt


def parse_frames_from_response(text: str) -> Tuple[int, int]:
    """
    Parse 'From frame X to frame Y' from model output.
    If parsing fails, return (0, 0).
    """
    # Main pattern: "From frame X to frame Y"
    m = re.search(
        r"from\s*frame\s*(\d+)\s*to\s*frame\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        s = int(m.group(1))
        e = int(m.group(2))
        if s > e:
            s, e = e, s
        return s, e

    # Fallback: "from X to Y"
    m = re.search(
        r"from\s*(\d+)\s*to\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        s = int(m.group(1))
        e = int(m.group(2))
        if s > e:
            s, e = e, s
        return s, e

    # Last fallback: first two integers
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        s = int(nums[0])
        e = int(nums[1])
        if s > e:
            s, e = e, s
        return s, e

    return 0, 0


# =========================
# 4. Main evaluation loop (MovieNet version)
# =========================

def run_eval(
    json_path: str,
    video_root: str,
    save_path: str,
    device: str = "cuda",
    position: str = "bottom_right",
    font_size: int = 40,
    color: str = "red",
):
    """
    Evaluate Qwen2-VL on MovieNet_Gemini_500_dataset.json

    Args:
        json_path: path to MovieNet_Gemini_500_dataset.json.
        video_root: folder containing original <movie_id>.mp4 (without numbers).
        save_path: output json (metrics + per-sample results).
        fps: original FPS used to convert frame index -> seconds.
             If <= 0, use each video's own FPS from OpenCV.
        sample_fps: sampling FPS for Qwen (only控制 Qwen downsample 的密度)。
        device: device for input tensors, e.g. "cuda" or "cuda:0".
        position/font_size/color: frame number 顯示位置與樣式。
    """
    # 1) Load MovieNet dataset (flat list)
    samples = load_movienet_json(json_path)

    # 2) Load model & processor
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"Loading model {model_name} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    # Temporary directory for annotated videos
    temp_dir = tempfile.mkdtemp(prefix="qwen_movienet_")

    all_results: List[Dict[str, Any]] = []

    # cache: movie_id -> (annotated_path, orig_fps, total_frames)
    video_cache: Dict[str, Tuple[str, float, int]] = {}

    try:
        # 3) Iterate over segments
        for sample in tqdm(samples, desc="Evaluating MovieNet"):
            mid = sample["movie_id"]
            query = sample["description"]

            orig_video_path = os.path.join(video_root, f"{mid}.mp4")
            if not os.path.exists(orig_video_path):
                print(f"[WARN] video not found: {orig_video_path}")
                result = {
                    "movie_id": mid,
                    "description": query,
                    "gt_start": 0.0,
                    "gt_end": 0.0,
                    "pred_start_frame": 0,
                    "pred_end_frame": 0,
                    "pred_start_sec": 0.0,
                    "pred_end_sec": 0.0,
                    "raw_response": "[VIDEO_NOT_FOUND]",
                }
                all_results.append(result)
                continue

            # 3a) 如果這部電影還沒被標號過，就整隻影片加 frame number
            if mid not in video_cache:
                annotated_path = os.path.join(temp_dir, f"{mid}_indexed.mp4")
                orig_fps, total_frames = annotate_and_save_video(
                    orig_video_path,
                    annotated_path,
                    position=position,
                    font_size=font_size,
                    color=color,
                )
                video_cache[mid] = (annotated_path, orig_fps, total_frames)
            else:
                annotated_path, orig_fps, total_frames = video_cache[mid]

            # GT: MovieNet 的 GT 是 frame index -> 轉成秒數
            gt_start_sec = sample["start_shot"]
            gt_end_sec = sample["end_shot"]

            # 3b) Build prompt (同 charades 版，不提 duration / fps)
            prompt = build_prompt(query, total_frames)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": annotated_path,
                            "fps": 200/total_frames
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_inputs = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_inputs],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                )

            # Strip prompt part
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            print(output_text)

            pred_start_frame, pred_end_frame = parse_frames_from_response(output_text)

            # 3c) Convert frame index -> seconds for IoU
            pred_start_sec = pred_start_frame / orig_fps
            pred_end_sec = pred_end_frame / orig_fps

            result = {
                "movie_id": mid,
                "description": query,
                "gt_start": gt_start_sec,    # seconds
                "gt_end": gt_end_sec,        # seconds
                "pred_start_frame": pred_start_frame,
                "pred_end_frame": pred_end_frame,
                "pred_start_sec": pred_start_sec,
                "pred_end_sec": pred_end_sec,
                "orig_fps": orig_fps,
                "total_frames": total_frames,
                "raw_response": output_text,
            }
            all_results.append(result)

        # 4) Compute metrics (mIoU + recalls) in seconds
        metrics = compute_metrics(all_results)
        metrics["num_samples"] = len(all_results)

        out_obj = {
            "metrics": metrics,
            "results": all_results,
        }
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print("=== MovieNet Evaluation done ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================
# 5. CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to MovieNet_Gemini_500_dataset.json",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Folder containing original <movie_id>.mp4 (without frame numbers).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="qwens_MovieNet_results.json",
        help="Where to save the output JSON (metrics + per-sample results).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inputs (model itself uses device_map='auto').",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="bottom_right",
        help="Position of the frame number annotation (default: bottom_right).",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=80,
        help="Font size of the frame number annotation.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="red",
        help="Color of the frame number annotation.",
    )
    args = parser.parse_args()

    run_eval(
        json_path=args.json_path,
        video_root=args.video_dir,
        save_path=args.output_json,
        device=args.device,
        position=args.position,
        font_size=args.font_size,
        color=args.color,
    )
