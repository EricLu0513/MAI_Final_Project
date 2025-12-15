import os
import json
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


# =========================
# JSON loading & utilities
# =========================

def load_movienet_segments(json_path: str) -> Dict[str, List[dict]]:
    """
    讀 MovieNet_Gemini_500_dataset.json，依 movie_id 分組。
    回傳: {movie_id: [segment_dict, ...]}
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


# =========================
# Video sampling
# =========================

def sample_frames_from_video(
    video_path: str,
    fps: float
) -> Tuple[List[Image.Image], List[float], float]:
    """
    以指定 FPS 從影片取樣。
    回傳:
      - frames: list of PIL Images
      - frame_times: 每張 sample frame 對應原始影片時間（秒）
      - orig_fps: 影片原始 FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = fps  # fallback

    frame_interval = max(int(round(orig_fps / fps)), 1)

    frames: List[Image.Image] = []
    frame_times: List[float] = []

    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
            frame_times.append(idx / orig_fps)
        idx += 1

    cap.release()
    return frames, frame_times, orig_fps


# =========================
# Text processing
# =========================

def split_description_into_sentences(desc: str) -> List[str]:
    """
    把英文 description 用句點切成短句。
    （很簡單的版本，需要更好可再自己換）
    """
    desc = desc.replace("\n", " ")
    parts = [p.strip() for p in desc.split(".") if p.strip()]
    return parts if parts else [desc.strip()]


# =========================
# CLIP scoring
# =========================

def compute_clip_scores_for_frames(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    frames: List[Image.Image],
    sentences: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    對每張 frame，計算與所有句子的 CLIP similarity，
    該 frame score = 與所有句子中的最大 similarity。
    回傳長度 N 的 numpy array。
    """
    model.eval()

    # 先 encode 所有文字
    with torch.no_grad():
        text_inputs = processor(
            text=sentences,
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    scores = []

    # 再分批 encode 影格
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i: i + batch_size]
        with torch.no_grad():
            inputs = processor(
                text=None,
                images=batch_frames,
                return_tensors="pt",
                padding=True,
            ).to(device)
            image_embeds = model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # cosine similarity: (B, D) x (D, M) -> (B, M)
            sim = image_embeds @ text_embeds.T
            # 每張圖取對所有句子的最大值
            max_sim, _ = sim.max(dim=1)
            scores.extend(max_sim.cpu().numpy().tolist())

    return np.array(scores, dtype=np.float32)


# =========================
# Coarse range selection
# =========================

def kadane_best_segment(arr: np.ndarray) -> Tuple[int, int, float]:
    """
    Kadane 演算法找最大總和子陣列。
    回傳 (start_idx, end_idx, max_sum)，0-based、包含 end。
    """
    best_sum = -float("inf")
    cur_sum = 0.0
    best_start = best_end = 0
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
    重複：
      1) 對目前區間做 normalize (mean=0, std=1)
      2) 用 Kadane 找最大總和子陣列
      3) 把目前區間縮小成那個子陣列

    直到：
      - 區間長度 <= max_len，或
      - 再也縮不小（避免無窮迴圈）

    回傳 (start_idx, end_idx) 為原本 scores 的 index。
    """
    if len(scores) == 0:
        return 0, 0

    cur_start = 0
    cur_end = len(scores) - 1

    while True:
        length = cur_end - cur_start + 1
        if length <= max_len:
            break

        sub = scores[cur_start: cur_end + 1]
        mean = float(sub.mean())
        std = float(sub.std())
        if std < 1e-6:
            # 幾乎全一樣，沒辦法再靠 normalize 縮小
            break
        z = (sub - mean) / std
        rel_s, rel_e, _ = kadane_best_segment(z)
        new_start = cur_start + rel_s
        new_end = cur_start + rel_e

        # 如果沒縮小就停
        if new_end - new_start + 1 >= length:
            break

        cur_start, cur_end = new_start, new_end

    return cur_start, cur_end


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
# Main
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
                        help="Sampling FPS for coarse CLIP stage.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device, e.g., cuda or cpu.")
    parser.add_argument("--max_len", type=int, default=60,
                        help="Max coarse range length (in sampled frames).")
    parser.add_argument("--clip_model_name", type=str,
                        default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 讀取 JSON，全部 segments
    segments_by_movie = load_movienet_segments(args.json_path)
    movie_ids = sorted(segments_by_movie.keys())
    print(f"Total movies in JSON: {len(movie_ids)}")

    # 2. 載 CLIP model
    print("Loading CLIP model:", args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    clip_model.eval()

    # 統計量
    total_segments = 0
    sum_iou = 0.0
    recall_03 = 0
    recall_05 = 0
    recall_07 = 0

    missing_video_count = 0

    # Optional: 存每筆結果
    all_results = []

    # 3. 對每個 movie 跑
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

        # 3a. 取樣影片（共用 frames 給該影片所有 query）
        frames, frame_times, orig_fps = sample_frames_from_video(
            video_path,
            fps=args.sample_fps
        )
        num_frames = len(frames)
        print(f"Sampled {num_frames} frames at {args.sample_fps} fps (orig_fps={orig_fps:.3f})")

        if num_frames == 0:
            print("[WARN] No frames sampled, skip this movie.")
            continue

        # 3b. 對每個 segment / query 做 coarse retrieval
        for seg in movie_segments:
            total_segments += 1
            desc = seg["description"]
            sentences = split_description_into_sentences(desc)

            gt_start_sec = seg["start_shot"]
            gt_end_sec = seg["end_shot"]

            # CLIP scores
            scores = compute_clip_scores_for_frames(
                model=clip_model,
                processor=clip_processor,
                device=device,
                frames=frames,
                sentences=sentences,
            )

            # iterative shrink to <= max_len frames
            pred_start_idx, pred_end_idx = iterative_best_range(
                scores,
                max_len=args.max_len
            )

            # mapping sampled frame index -> time (用 sample_fps)
            pred_start_sec = pred_start_idx / args.sample_fps
            pred_end_sec = pred_end_idx / args.sample_fps

            iou = temporal_iou(
                pred_start_sec,
                pred_end_sec,
                gt_start_sec,
                gt_end_sec,
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
                    "pred_start_idx": int(pred_start_idx),
                    "pred_end_idx": int(pred_end_idx),
                    "pred_start_sec": pred_start_sec,
                    "pred_end_sec": pred_end_sec,
                    "iou": iou,
                }
            )

    # 4. 統計結果
    print("\n" + "=" * 80)
    print("Finished coarse-only evaluation on dataset.")
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

    print(f"mIoU:         {miou:.4f}")
    print(f"Recall@0.3:   {r03:.4f}")
    print(f"Recall@0.5:   {r05:.4f}")
    print(f"Recall@0.7:   {r07:.4f}")

    # 如果你想存 per-sample 結果，可以在這裡 dump 一份 json
    out_path = "clip_coarse_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "miou": miou,
                "recall_0.3": r03,
                "recall_0.5": r05,
                "recall_0.7": r07,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
