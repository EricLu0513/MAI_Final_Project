import os
import re
import json
import argparse
import shutil
import tempfile
from typing import List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# =========================
# JSON & basic utilities
# =========================

def load_movienet_segments(json_path: str) -> Dict[str, List[dict]]:
    """
    Load MovieNet_Gemini_500_dataset.json and group segments by movie_id.
    Returns dict: {movie_id: [segment, ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data["segments"]
    by_movie: Dict[str, List[dict]] = {}
    for seg in segments:
        mid = seg["movie_id"]
        by_movie.setdefault(mid, []).append(seg)
    return by_movie


def build_video_path(video_dir: str, movie_id: str, ext: str = ".mp4") -> str:
    """
    給 movie_id 回傳對應影片路徑，預設檔名為 {movie_id}.mp4
    """
    return os.path.join(video_dir, movie_id + ext)


def split_description_into_sentences(desc: str) -> List[str]:
    """
    Very simple split by period '.' for English descriptions.
    """
    desc = desc.replace("\n", " ")
    parts = [p.strip() for p in desc.split(".") if p.strip()]
    return parts if parts else [desc.strip()]


# =========================
# Sampling video for CLIP coarse stage
# =========================

def sample_video_frames(
    video_path: str,
    sample_fps: float,
) -> Tuple[List[Image.Image], List[float], float, float]:
    """
    Sample video frames at sample_fps.
    Returns:
      - frames_pil: list of PIL Images
      - frame_times: list of time (sec) of each sampled frame
      - orig_fps: original fps of video
      - duration: video duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = sample_fps  # fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if total_frames > 0 else 0.0

    # how many original frames per one sampled frame
    frame_interval = max(1, int(round(orig_fps / sample_fps)))

    frames_pil: List[Image.Image] = []
    frame_times: List[float] = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames_pil.append(pil_img)
            frame_times.append(frame_idx / orig_fps)
        frame_idx += 1

    cap.release()
    return frames_pil, frame_times, orig_fps, duration


# =========================
# CLIP scoring
# =========================

def compute_clip_scores_for_frames(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    frames_pil: List[Image.Image],
    sentences: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    For each frame, compute CLIP similarity to a list of sentences,
    and take the max similarity as the frame score.
    Returns array of shape (N_frames,).
    """
    clip_model.eval()

    # Encode all sentences
    with torch.no_grad():
        text_inputs = clip_processor(
            text=sentences,
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_embeds = clip_model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # (M, D)

    scores: List[float] = []
    num_frames = len(frames_pil)

    # Encode frames in batches
    for i in range(0, num_frames, batch_size):
        batch_frames = frames_pil[i : i + batch_size]
        with torch.no_grad():
            img_inputs = clip_processor(
                text=None,
                images=batch_frames,
                return_tensors="pt",
                padding=True,
            ).to(device)
            img_embeds = clip_model.get_image_features(**img_inputs)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)  # (B, D)

            # cosine similarity (B, M)
            sims = img_embeds @ text_embeds.T
            # for each frame, take max over all sentences
            max_sims, _ = sims.max(dim=1)
            scores.extend(max_sims.cpu().tolist())

    return np.array(scores, dtype=np.float32)


# =========================
# Iterative normalize + max-subarray (coarse)
# =========================

def kadane_best_segment(arr: np.ndarray) -> Tuple[int, int, float]:
    """
    Kadane's algorithm to find max-sum subarray.
    Returns (start_idx, end_idx, max_sum), end inclusive.
    """
    best_sum = -float("inf")
    cur_sum = 0.0
    best_start = 0
    best_end = 0
    cur_start = 0

    for i, x in enumerate(arr):
        if cur_sum <= 0:
            cur_sum = x
            cur_start = i
        else:
            cur_sum += x

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_start = cur_start
            best_end = i

    return best_start, best_end, best_sum


def iterative_best_range(scores: np.ndarray, max_len: int = 60) -> Tuple[int, int]:
    """
    Repeatedly:
      1) z-normalize current segment,
      2) run Kadane to find best subarray,
      3) shrink current segment to that subarray,
    until length <= max_len or no further shrink.

    Returns (start_idx, end_idx) in original scores indexing.
    """
    if len(scores) == 0:
        return 0, 0

    cur_start = 0
    cur_end = len(scores) - 1

    while True:
        length = cur_end - cur_start + 1
        if length <= max_len:
            break

        sub = scores[cur_start : cur_end + 1]
        mean = float(sub.mean())
        std = float(sub.std())
        if std < 1e-6:
            # all scores very similar; no effect from normalization
            break

        z = (sub - mean) / std
        rel_s, rel_e, _ = kadane_best_segment(z)
        new_start = cur_start + rel_s
        new_end = cur_start + rel_e
        new_len = new_end - new_start + 1

        # if not actually shrinking, stop
        if new_len >= length:
            break

        cur_start, cur_end = new_start, new_end

    return cur_start, cur_end


# =========================
# Qwen2-VL fine stage helpers
# =========================

def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a TTF font; fallback to default bitmap font if unavailable.
    """
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
    Draw frame index text onto frame using PIL.
    position: top_left, top_right, bottom_left, bottom_right, center
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
        x, y = margin, margin

    if position in ["bottom_left", "bottom_right"]:
        y -= text_height / 3

    draw.text((x, y), text, font=font, fill=color)
    frame_bgr_out = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame_bgr_out


def extract_subvideo_with_index(
    src_path: str,
    dst_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    position: str = "bottom_right",
    font_size: int = 40,
    color: str = "red",
) -> Tuple[float, int]:
    """
    Cut [start_sec, end_sec] from src video, sample at sample_fps,
    overlay frame index text, and save to dst_path.
    Returns (sample_fps, num_frames_in_subvideo).
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = sample_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if total_frames > 0 else 0.0

    start_sec = max(0.0, start_sec)
    end_sec = max(start_sec, min(end_sec, duration))

    start_frame = int(round(start_sec * orig_fps))
    end_frame = int(round(end_sec * orig_fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, sample_fps, (width, height))

    sample_interval = max(1, int(round(orig_fps / sample_fps)))
    frame_idx = 0
    sub_frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_orig = start_frame

    while cur_orig <= end_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if (cur_orig - start_frame) % sample_interval == 0:
            annotated = annotate_frame_with_index(
                frame_bgr,
                text=str(sub_frame_idx),
                position=position,
                font_size=font_size,
                color=color,
            )
            out.write(annotated)
            sub_frame_idx += 1

        cur_orig += 1
        frame_idx += 1

    cap.release()
    out.release()
    return sample_fps, sub_frame_idx


def build_qwen_prompt(query: str, total_frames: int) -> str:
    """
    Build English prompt for Qwen2-VL.
    """
    prompt = (
        "The red numbers on each frame represent the frame number.\n"
        f"The total number of frame is {total_frames}\n"
        f"In this video, during which frames can we see '{query}'? "
        "Answer in the format: 'From Frame x to Frame y'"
    )
    return prompt


def parse_frames_from_response(text: str) -> Tuple[int, int]:
    """
    Parse 'From Frame x to Frame y' (or variants) from model output text.
    Returns (start_frame, end_frame) within subvideo index.
    """
    m = re.search(r"from\s*frame\s*(\d+)\s*to\s*frame\s*(\d+)", text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        e = int(m.group(2))
        if s > e:
            s, e = e, s
        return s, e

    m = re.search(r"from\s*(\d+)\s*to\s*(\d+)", text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        e = int(m.group(2))
        if s > e:
            s, e = e, s
        return s, e

    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        s = int(nums[0])
        e = int(nums[1])
        if s > e:
            s, e = e, s
        return s, e

    return 0, 0


def refine_with_qwen(
    video_path: str,
    coarse_start_sec: float,
    coarse_end_sec: float,
    query: str,
    qwen_model: Qwen2VLForConditionalGeneration,
    qwen_processor: AutoProcessor,
    device: torch.device,
    fine_sample_fps: float = 1.0,
    max_frames_for_qwen: int = 60,
    position: str = "bottom_right",
    font_size: int = 40,
    color: str = "red",
) -> Tuple[float, float, str]:
    """
    Use Qwen2-VL to refine temporal range inside [coarse_start_sec, coarse_end_sec].

    這裡會先用 fine_sample_fps 裁一次 subvideo，
    如果實際 frame 數 > max_frames_for_qwen，就自動下修 fps 再裁一次，
    以避免 Qwen 的輸入太長。

    Returns:
      - final_start_sec (in original video time)
      - final_end_sec
      - raw_text_output from Qwen
    """
    temp_dir = tempfile.mkdtemp(prefix="coarse_to_qwen_")
    try:
        subvideo_path = os.path.join(temp_dir, "subvideo.mp4")

        # 1) 先用原本 fine_sample_fps 裁一次
        sub_fps, total_frames = extract_subvideo_with_index(
            video_path,
            subvideo_path,
            coarse_start_sec,
            coarse_end_sec,
            sample_fps=fine_sample_fps,
            position=position,
            font_size=font_size,
            color=color,
        )

        # 2) 如果 frame 太多，自動下修 fps 再裁一次
        if total_frames > max_frames_for_qwen:
            duration = coarse_end_sec - coarse_start_sec
            duration = max(duration, 1e-6)
            adjusted_fps = max_frames_for_qwen / duration
            adjusted_fps = min(adjusted_fps, fine_sample_fps)
            adjusted_fps = max(adjusted_fps, 0.25)

            sub_fps, total_frames = extract_subvideo_with_index(
                video_path,
                subvideo_path,
                coarse_start_sec,
                coarse_end_sec,
                sample_fps=adjusted_fps,
                position=position,
                font_size=font_size,
                color=color,
            )

        prompt = build_qwen_prompt(query, total_frames)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": subvideo_path,
                        "fps": sub_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_inputs = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text_inputs],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        qwen_model.eval()
        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=64,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        sub_start_frame, sub_end_frame = parse_frames_from_response(output_text)
        sub_start_frame = max(0, sub_start_frame)
        sub_end_frame = max(sub_start_frame, sub_end_frame)

        final_start_sec = coarse_start_sec + sub_start_frame / sub_fps
        final_end_sec = coarse_start_sec + sub_end_frame / sub_fps

        return final_start_sec, final_end_sec, output_text

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================
# IoU & metrics
# =========================

def temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """
    計算兩個 [start, end] 區間的 IoU（時間單位：秒）。
    """
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    if gt_end < gt_start:
        gt_start, gt_end = gt_end, gt_start

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0.0, inter_end - inter_start)

    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = max(1e-6, union_end - union_start)

    return inter / union


# =========================
# Main pipeline over full dataset
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing all movie videos (movie_id.mp4).")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to MovieNet_Gemini_500_dataset.json.")
    parser.add_argument("--video_ext", type=str, default=".mp4",
                        help="Video file extension, default .mp4.")
    parser.add_argument("--sample_fps", type=float, default=1.0,
                        help="Coarse sampling FPS for CLIP.")
    parser.add_argument("--fine_fps", type=float, default=1.0,
                        help="Subvideo FPS for Qwen.")
    parser.add_argument("--max_len", type=int, default=60,
                        help="Maximum coarse range length (in sampled frames) "
                             "and max frames for Qwen subvideo.")
    parser.add_argument("--expand_before", type=float, default=2.0,
                        help="Expand seconds before coarse start for Qwen.")
    parser.add_argument("--expand_after", type=float, default=2.0,
                        help="Expand seconds after coarse end for Qwen.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--clip_model_name", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--qwen_model_name", type=str,
                        default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--position", type=str, default="bottom_right")
    parser.add_argument("--font_size", type=int, default=40)
    parser.add_argument("--color", type=str, default="red")
    parser.add_argument("--output_json", type=str,
                        default="qwen2_MovieNet_coarse_to_fine_results.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) load segments for all movies
    segments_by_movie = load_movienet_segments(args.json_path)
    movie_ids = sorted(segments_by_movie.keys())
    print(f"Total movies in JSON: {len(movie_ids)}")

    # 2) load CLIP
    print("Loading CLIP model:", args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    # 3) load Qwen2-VL
    print("Loading Qwen2-VL model:", args.qwen_model_name)
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.qwen_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_name)

    # metrics
    total_segments = 0
    sum_iou = 0.0
    recall_03 = 0
    recall_05 = 0
    recall_07 = 0
    missing_video_count = 0

    all_results = []

    # 4) loop over movies
    for movie_id in movie_ids:
        video_path = build_video_path(args.video_dir, movie_id, args.video_ext)
        if not os.path.isfile(video_path):
            print(f"[WARN] Video for movie_id {movie_id} not found at {video_path}, skip all its segments.")
            missing_video_count += 1
            continue

        movie_segments = segments_by_movie[movie_id]
        print("=" * 80)
        print(f"Processing movie_id={movie_id}, #segments={len(movie_segments)}")
        print(f"Video path: {video_path}")

        # 4a) coarse sampling of the whole video (shared for all queries)
        frames_pil, frame_times, orig_fps, duration = sample_video_frames(
            video_path,
            sample_fps=args.sample_fps,
        )
        num_frames = len(frames_pil)
        print(f"Sampled {num_frames} frames at {args.sample_fps} fps "
              f"(orig_fps={orig_fps:.3f}, duration={duration:.2f}s).")

        if num_frames == 0:
            print("[WARN] No frames sampled, skip this movie.")
            continue

        # 4b) for each segment / query
        for seg in movie_segments:
            total_segments += 1
            desc = seg["description"]
            sentences = split_description_into_sentences(desc)

            gt_start_sec = seg["start_shot"]
            gt_end_sec = seg["end_shot"]

            print("\n" + "-" * 80)
            print(f"movie_id={movie_id}, segment #{total_segments}")
            print("Query:", desc)
            print(f"(time {gt_start_sec:.2f}s -> {gt_end_sec:.2f}s)")

            # coarse CLIP scores
            scores = compute_clip_scores_for_frames(
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                frames_pil=frames_pil,
                sentences=sentences,
            )

            # coarse range (iterative shrink)
            coarse_start_idx, coarse_end_idx = iterative_best_range(
                scores,
                max_len=args.max_len,
            )
            coarse_len = coarse_end_idx - coarse_start_idx + 1
            coarse_start_sec = frame_times[coarse_start_idx]
            coarse_end_sec = frame_times[coarse_end_idx]

            # expand for Qwen
            coarse_start_sec_exp = max(0.0, coarse_start_sec - args.expand_before)
            coarse_end_sec_exp = min(duration, coarse_end_sec + args.expand_after)

            print(f"Coarse idx: {coarse_start_idx} -> {coarse_end_idx} (len={coarse_len})")
            print(f"Coarse time: {coarse_start_sec:.2f}s -> {coarse_end_sec:.2f}s "
                  f"(expanded: {coarse_start_sec_exp:.2f}s -> {coarse_end_sec_exp:.2f}s)")

            # fine refinement with Qwen2-VL
            final_start_sec, final_end_sec, qwen_text = refine_with_qwen(
                video_path=video_path,
                coarse_start_sec=coarse_start_sec_exp,
                coarse_end_sec=coarse_end_sec_exp,
                query=desc,
                qwen_model=qwen_model,
                qwen_processor=qwen_processor,
                device=device,
                fine_sample_fps=args.fine_fps,
                max_frames_for_qwen=args.max_len,
                position=args.position,
                font_size=args.font_size,
                color=args.color,
            )

            print("Qwen output:", qwen_text)
            print(f"Final refined time: {final_start_sec:.2f}s -> {final_end_sec:.2f}s")

            # IoU on refined result
            iou = temporal_iou(
                pred_start=final_start_sec,
                pred_end=final_end_sec,
                gt_start=gt_start_sec,
                gt_end=gt_end_sec,
            )
            sum_iou += iou
            if iou >= 0.3:
                recall_03 += 1
            if iou >= 0.5:
                recall_05 += 1
            if iou >= 0.7:
                recall_07 += 1

            all_results.append(
                {
                    "movie_id": movie_id,
                    "description": desc,
                    "gt_start_sec": gt_start_sec,
                    "gt_end_sec": gt_end_sec,
                    "coarse_start_idx": int(coarse_start_idx),
                    "coarse_end_idx": int(coarse_end_idx),
                    "coarse_start_sec": float(coarse_start_sec),
                    "coarse_end_sec": float(coarse_end_sec),
                    "coarse_start_sec_exp": float(coarse_start_sec_exp),
                    "coarse_end_sec_exp": float(coarse_end_sec_exp),
                    "final_start_sec": float(final_start_sec),
                    "final_end_sec": float(final_end_sec),
                    "iou": float(iou),
                    "qwen_output": qwen_text,
                }
            )

    # 5) compute metrics
    print("\n" + "=" * 80)
    print("Finished coarse-to-fine evaluation on dataset.")
    print(f"Total segments evaluated: {total_segments}")
    if missing_video_count > 0:
        print(f"Movies missing video files: {missing_video_count}")

    if total_segments > 0:
        miou = sum_iou / total_segments
        r03 = recall_03 / total_segments
        r05 = recall_05 / total_segments
        r07 = recall_07 / total_segments
    else:
        miou = r03 = r05 = r07 = 0.0

    print(f"mIoU (refined):  {miou:.4f}")
    print(f"Recall@0.3:      {r03:.4f}")
    print(f"Recall@0.5:      {r05:.4f}")
    print(f"Recall@0.7:      {r07:.4f}")

    # 6) save all results
    out_obj = {
        "miou": miou,
        "recall_0.3": r03,
        "recall_0.5": r05,
        "recall_0.7": r07,
        "total_segments": total_segments,
        "missing_videos": missing_video_count,
        "results": all_results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed results to: {args.output_json}")


if __name__ == "__main__":
    main()
