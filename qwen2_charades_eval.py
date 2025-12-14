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

def load_test_json(path: str) -> List[Dict[str, Any]]:
    """
    Load original test.json.

    Each item format:
        [video_id, duration, [start_time, end_time], query_sentence]
    Times are in seconds.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data: List[Dict[str, Any]] = []
    for item in raw:
        vid, duration, (gt_start, gt_end), query = item
        data.append(
            {
                "video_id": vid,
                "duration": float(duration),
                "gt_start": float(gt_start),
                "gt_end": float(gt_end),
                "query": query,
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
    font_size: int = 40,
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
    sample_fps: float = 1.0,
) -> Tuple[float, int]:
    """
    Coarse stage:
    - Read original video from src_path, downsample to sample_fps,
      overlay frame index (bottom-right, from 0..N-1), and save annotated video to dst_path.
    - Optionally resize frames by `resize_scale` (0<scale<=1) while preserving aspect ratio.
    Returns:
        (orig_fps, total_sampled_frames)
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file: {src_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = sample_fps

    # 時間下採樣比例
    sampled_rate = max(1, int(round(fps / sample_fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, sample_fps, (width, height))

    frame_idx = 0        # 原始影片的 frame index
    frame_count = 0      # 下採樣後的 frame index (0..N-1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sampled_rate == 0:
            """frame = annotate_frame_with_index(
                frame,
                str(frame_count),
                position=position,
                font_size=font_size,
                color=color,
            )"""
            frame_count += 1
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    return fps, frame_count



# =========================
# 3. Prompt & parsing
# =========================

def build_prompt(query: str, total_frames: int) -> str:
    """
    English prompt.

    - Mention bottom-right number is the frame index.
    - Ask model to answer 'From frame X to frame Y'.
    - Do NOT mention duration / fps.
    """
    prompt = (
        "The red numbers on each frame represent the frame number.\n"
        f"The total number of frame is {total_frames}\n"
        f"In this video, during which frames can we see {query}? Answer in the format: 'From Frame x to Frame y'"
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
# 4. Main evaluation loop
# =========================

def run_eval(
    data_path: str,
    video_root: str,
    save_path: str,
    sample_fps: float = 1.0,
    device: str = "cuda",
    position: str = "bottom_right",
    font_size: int = 40,
    color: str = "red",
    resize_scale: float = 1.0,
):
    """
    Evaluate Qwen2-VL on test.json.

    Args:
        data_path: path to test.json ([vid, duration, [st, ed], query]).
        video_root: folder containing original <video_id>.mp4 (without numbers).
        save_path: output json (metrics + per-sample results).
        sample_fps: FPS for Qwen video sampling (e.g., 1.0 for speed).
        device: device for input tensors, e.g. "cuda" or "cuda:0".
        position/font_size/color: for frame number overlay at bottom-right.
    """
    # 1) Load dataset
    samples = load_test_json(data_path)

    # 2) Load model & processor (with flash attention)
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
    temp_dir = tempfile.mkdtemp(prefix="qwen_charades_")

    all_results: List[Dict[str, Any]] = []

    try:
        # 3) Iterate over samples
        for sample in tqdm(samples, desc="Evaluating"):
            vid = sample["video_id"]
            orig_video_path = os.path.join(video_root, f"{vid}.mp4")

            if not os.path.exists(orig_video_path):
                print(f"[WARN] video not found: {orig_video_path}")
                result = {
                    "video_id": vid,
                    "query": sample["query"],
                    "gt_start": sample["gt_start"],
                    "gt_end": sample["gt_end"],
                    "duration": sample["duration"],
                    "pred_start_frame": 0,
                    "pred_end_frame": 0,
                    "pred_start_sec": 0.0,
                    "pred_end_sec": 0.0,
                    "raw_response": "[VIDEO_NOT_FOUND]",
                }
                all_results.append(result)
                continue

            # 3a) Create annotated video with bottom-right frame index
            annotated_path = os.path.join(temp_dir, f"{vid}_indexed.mp4")
            orig_fps, total_frames = annotate_and_save_video(
                orig_video_path,
                annotated_path,
                position=position,
                font_size=font_size,
                color=color,
                sample_fps=sample_fps
            )
            


            # 3b) Build prompt (English, mention bottom-right frame number)
            prompt = build_prompt(sample["query"], total_frames)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": annotated_path,
                            #"max_pixels": 144 * 144,
                            #"min_pixels": 144 * 144,
                            "fps": sample_fps,  # sampling fps for Qwen (only affects how many frames it sees)
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
            pred_start_sec = pred_start_frame / sample_fps
            pred_end_sec = pred_end_frame / sample_fps


            result = {
                "video_id": vid,
                "query": sample["query"],
                "gt_start": sample["gt_start"],   # seconds
                "gt_end": sample["gt_end"],       # seconds
                "duration": sample["duration"],   # seconds
                "pred_start_frame": pred_start_frame,
                "pred_end_frame": pred_end_frame,
                "pred_start_sec": pred_start_sec,
                "pred_end_sec": pred_end_sec,
                "orig_fps": orig_fps,
                "sample_fps": sample_fps,
                "total_frames": total_frames,
                "raw_response": output_text,
            }
            all_results.append(result)

        # 4) Compute metrics (mIoU + recalls) using *seconds*
        metrics = compute_metrics(all_results)
        metrics["num_samples"] = len(all_results)

        # 5) Save output JSON (fix makedirs when save_path has no directory)
        out_obj = {
            "metrics": metrics,
            "results": all_results,
        }
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print("=== Evaluation done ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    finally:
        # Clean up temporary annotated videos
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================
# 5. CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="test.json",
        help="Path to original test.json ([vid, duration, [st, ed], query]).",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Folder containing original <video_id>.mp4 (without frame numbers).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="qwen_test_results.json",
        help="Where to save the output JSON (metrics + per-sample results).",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="FPS for Qwen video sampling (smaller -> fewer frames, faster).",
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
    parser.add_argument(
        "--resize_scale",
        type=float,
        default=1.0,
        help="resize scale of each video relative to the input resolution",
    )
    args = parser.parse_args()

    run_eval(
        data_path=args.data_path,
        video_root=args.video_root,
        save_path=args.save_path,
        sample_fps=args.sample_fps,
        device=args.device,
        position=args.position,
        font_size=args.font_size,
        color=args.color,
        resize_scale=args.resize_scale
    )
