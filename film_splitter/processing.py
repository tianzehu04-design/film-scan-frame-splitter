from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class CropBox:
    """A crop rectangle in original-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    order: int
    enabled: bool = True

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def clamp(self, image_width: int, image_height: int) -> "CropBox":
        x = max(0, min(int(round(self.x)), image_width - 1))
        y = max(0, min(int(round(self.y)), image_height - 1))
        right = max(x + 1, min(int(round(self.right)), image_width))
        bottom = max(y + 1, min(int(round(self.bottom)), image_height))
        return CropBox(x, y, right - x, bottom - y, int(self.order), bool(self.enabled))


def open_uploaded_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    image.load()
    if image.mode not in {"RGB", "RGBA", "L", "I;16", "I;16B", "I;16L"}:
        image = image.convert("RGB")
    return image


def make_preview(image: Image.Image, max_side: int = 1800) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height)))
    if scale >= 1:
        preview = image.copy()
    else:
        preview = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    return preview, scale


def pil_to_cv_gray(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return clahe.apply(blurred)


def detect_orientation(boxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int]) -> str:
    height, width = image_shape[:2]
    if len(boxes) >= 2:
        centers_x = np.array([x + w / 2 for x, _, w, _ in boxes])
        centers_y = np.array([y + h / 2 for _, y, _, h in boxes])
        return "horizontal" if np.ptp(centers_x) >= np.ptp(centers_y) else "vertical"
    return "horizontal" if width >= height else "vertical"


def _merge_overlapping_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not rects:
        return []

    def iou(a, b) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for rect in rects:
        if all(iou(rect, existing) < 0.35 for existing in kept):
            kept.append(rect)
    return kept


def _contour_frame_candidates(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    norm = normalize_gray(gray)

    # Combine thresholded tonal regions with strong edges. Film frames can be
    # bright positives, dark negatives, or low-contrast bordered areas, so using
    # both signals is more tolerant than a single global threshold.
    _, otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(norm, 40, 130)
    combined = cv2.bitwise_or(otsu, edges)

    kernel_major = max(9, int(min(width, height) * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_major, kernel_major))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = width * height
    min_area = image_area * 0.006
    max_area = image_area * 0.85
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if w < width * 0.08 and h < height * 0.08:
            continue
        aspect = w / float(h)
        if aspect < 0.25 or aspect > 6.5:
            continue

        contour_area = cv2.contourArea(contour)
        fill_ratio = contour_area / float(area) if area else 0
        if fill_ratio < 0.08:
            continue
        candidates.append((x, y, w, h))

    return _merge_overlapping_rects(candidates)


def _projection_fallback(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = gray.shape
    norm = normalize_gray(gray)
    edges = cv2.Canny(norm, 35, 120)

    orientation = "horizontal" if width >= height else "vertical"
    if orientation == "horizontal":
        profile = edges.mean(axis=0)
        axis_len = width
        cross_len = height
    else:
        profile = edges.mean(axis=1)
        axis_len = height
        cross_len = width

    smooth_size = max(11, int(axis_len * 0.015) | 1)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_size, 1), 0).ravel()
    threshold = max(float(np.percentile(profile, 55)), float(profile.mean() * 0.75))
    active = profile > threshold

    segments: list[tuple[int, int]] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            if idx - start > axis_len * 0.04:
                segments.append((start, idx))
            start = None
    if start is not None and axis_len - start > axis_len * 0.04:
        segments.append((start, axis_len))

    rects: list[tuple[int, int, int, int]] = []
    margin = int(cross_len * 0.05)
    for start, end in segments:
        if orientation == "horizontal":
            rects.append((start, margin, end - start, max(1, height - 2 * margin)))
        else:
            rects.append((margin, start, max(1, width - 2 * margin), end - start))
    return rects


def _close_small_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    active = active.astype(bool).copy()
    start = None
    for idx, value in enumerate(active):
        if not value and start is None:
            start = idx
        elif value and start is not None:
            if idx - start <= max_gap:
                active[start:idx] = True
            start = None
    return active


def _segments_from_active(active: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_length:
                segments.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_length:
        segments.append((start, len(active)))
    return segments


def _split_evenly(length: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, int(parts))
    return [
        (round(index * length / parts), round((index + 1) * length / parts))
        for index in range(parts)
    ]


def _auto_lanes(crop: np.ndarray, orientation: str, max_lanes: int = 4) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        white_profile = (crop > 248).mean(axis=1)
        total = crop.shape[0]
    else:
        white_profile = (crop > 248).mean(axis=0)
        total = crop.shape[1]

    smooth_size = max(9, int(total * 0.02) | 1)
    white_smoothed = cv2.GaussianBlur(white_profile.reshape(1, -1), (smooth_size, 1), 0).ravel()
    white_gaps = _segments_from_active(
        white_smoothed > 0.78,
        min_length=max(20, int(total * 0.06)),
    )
    white_gaps = [
        (start, end)
        for start, end in white_gaps
        if start > total * 0.08 and end < total * 0.92
    ]
    if white_gaps:
        lanes: list[tuple[int, int]] = []
        cursor = 0
        for start, end in white_gaps:
            if start - cursor >= total * 0.12:
                lanes.append((cursor, start))
            cursor = end
        if total - cursor >= total * 0.12:
            lanes.append((cursor, total))
        if 1 <= len(lanes) <= max_lanes:
            return lanes

    if orientation == "horizontal":
        profile = (crop < 245).mean(axis=1)
        total = crop.shape[0]
    else:
        profile = (crop < 245).mean(axis=0)
        total = crop.shape[1]

    smooth_size = max(9, int(total * 0.025) | 1)
    smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_size, 1), 0).ravel()
    active = smoothed > max(0.08, float(np.percentile(smoothed, 35)))
    active = _close_small_gaps(active, max(2, int(total * 0.03)))
    segments = _segments_from_active(active, min_length=max(20, int(total * 0.14)))

    if 1 <= len(segments) <= max_lanes:
        return segments
    return [(0, total)]


def _lanes_for_crop(crop: np.ndarray, orientation: str, lane_count: int) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        length = crop.shape[0]
    else:
        length = crop.shape[1]
    if lane_count <= 0:
        return _auto_lanes(crop, orientation)
    return [(0, length)] if lane_count == 1 else _split_evenly(length, lane_count)


def _strip_bounds(gray: np.ndarray) -> tuple[int, int, int, int]:
    height, width = gray.shape
    border = np.concatenate(
        [
            gray[: max(1, height // 30), :].ravel(),
            gray[-max(1, height // 30) :, :].ravel(),
            gray[:, : max(1, width // 30)].ravel(),
            gray[:, -max(1, width // 30) :].ravel(),
        ]
    )
    background = float(np.median(border))
    diff = cv2.absdiff(gray, np.full_like(gray, int(round(background))))
    threshold = max(10, int(np.percentile(diff, 72)))
    mask = (diff > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(7, width // 80), max(7, height // 80)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, width, height)

    min_area = width * height * 0.003
    rects = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= min_area]
    if not rects:
        rects = [cv2.boundingRect(max(contours, key=cv2.contourArea))]

    x1 = min(rect[0] for rect in rects)
    y1 = min(rect[1] for rect in rects)
    x2 = max(rect[0] + rect[2] for rect in rects)
    y2 = max(rect[1] + rect[3] for rect in rects)
    margin_x = max(2, int(width * 0.01))
    margin_y = max(2, int(height * 0.01))
    x = max(0, x1 - margin_x)
    y = max(0, y1 - margin_y)
    right = min(width, x2 + margin_x)
    bottom = min(height, y2 + margin_y)
    if (right - x) * (bottom - y) < width * height * 0.1:
        return (0, 0, width, height)
    return (x, y, right - x, bottom - y)


def _dominant_gray_value(values: np.ndarray) -> int:
    hist, bin_edges = np.histogram(values.astype(np.uint8), bins=64, range=(0, 256))
    index = int(np.argmax(hist))
    return int(round((bin_edges[index] + bin_edges[index + 1]) / 2))


def _largest_active_span(profile: np.ndarray, threshold: float, fallback_length: int) -> tuple[int, int]:
    active = profile > threshold
    active = _close_small_gaps(active, max(2, fallback_length // 40))
    segments = _segments_from_active(active, min_length=max(2, fallback_length // 30))
    if not segments:
        return (0, fallback_length)
    return max(segments, key=lambda segment: segment[1] - segment[0])


def _regularize_centers(
    centers: list[float],
    frame_axis: int,
    axis_len: int,
    expected_count: int = 0,
) -> list[float]:
    centers = sorted(centers)
    if expected_count > 0 and len(centers) >= 2:
        return np.linspace(centers[0], centers[-1], expected_count).tolist()
    if expected_count > 0 and len(centers) == 1:
        return [centers[0] + (index - expected_count // 2) * frame_axis for index in range(expected_count)]
    if len(centers) < 2:
        return centers

    diffs = np.diff(centers)
    pitch = float(np.median(diffs))
    if pitch <= max(1, frame_axis * 0.5):
        return centers

    regular = [centers[0]]
    current = centers[0]
    while current + pitch <= centers[-1] + pitch * 0.35:
        current += pitch
        regular.append(current)

    # Extend across the scanned strip only when the inferred full frame still fits.
    while regular and regular[0] - pitch - frame_axis / 2 >= 0:
        regular.insert(0, regular[0] - pitch)
    while regular and regular[-1] + pitch + frame_axis / 2 <= axis_len:
        regular.append(regular[-1] + pitch)
    return regular


def _normalize_rects_to_common_aspect(
    rects: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
    orientation: str,
    aspect_mode: str,
) -> list[tuple[int, int, int, int]]:
    if not rects:
        return rects

    landscape = aspect_mode != "Portrait 2:3"
    if orientation == "vertical":
        common_width = int(round(float(np.median([rect[2] for rect in rects]))))
        if landscape:
            common_width = max(3, int(round(common_width / 3)) * 3)
            common_height = common_width * 2 // 3
        else:
            common_width = max(2, int(round(common_width / 2)) * 2)
            common_height = common_width * 3 // 2
    else:
        common_height = int(round(float(np.median([rect[3] for rect in rects]))))
        if landscape:
            common_height = max(2, int(round(common_height / 2)) * 2)
            common_width = common_height * 3 // 2
        else:
            common_height = max(3, int(round(common_height / 3)) * 3)
            common_width = common_height * 2 // 3

    common_width = max(1, common_width)
    common_height = max(1, common_height)
    normalized: list[tuple[int, int, int, int]] = []
    for x, y, w, h in rects:
        center_x = x + w / 2
        center_y = y + h / 2
        nx = int(round(center_x - common_width / 2))
        ny = int(round(center_y - common_height / 2))
        nx = max(0, min(nx, image_width - common_width))
        ny = max(0, min(ny, image_height - common_height))
        normalized.append((nx, ny, common_width, common_height))
    return normalized


def _dedupe_same_lane_rects(
    rects: list[tuple[int, int, int, int]],
    orientation: str,
) -> list[tuple[int, int, int, int]]:
    if len(rects) < 2:
        return rects

    if orientation == "vertical":
        lane_key_index = 0
        axis_index = 1
        axis_size_index = 3
        lane_size_index = 2
    else:
        lane_key_index = 1
        axis_index = 0
        axis_size_index = 2
        lane_size_index = 3

    rects = sorted(rects, key=lambda rect: (rect[lane_key_index], rect[axis_index]))
    lanes: list[list[tuple[int, int, int, int]]] = []
    for rect in rects:
        center = rect[lane_key_index] + rect[lane_size_index] / 2
        placed = False
        for lane in lanes:
            lane_center = np.median([item[lane_key_index] + item[lane_size_index] / 2 for item in lane])
            lane_width = np.median([item[lane_size_index] for item in lane])
            if abs(center - lane_center) <= lane_width * 0.6:
                lane.append(rect)
                placed = True
                break
        if not placed:
            lanes.append([rect])

    kept: list[tuple[int, int, int, int]] = []
    allow_gap_fill = len(lanes) > 1
    for lane in lanes:
        lane = sorted(lane, key=lambda rect: rect[axis_index])
        lane_kept: list[tuple[int, int, int, int]] = []
        min_distance = max(1, int(round(np.median([rect[axis_size_index] for rect in lane]) * 0.72)))
        for rect in lane:
            if lane_kept and rect[axis_index] - lane_kept[-1][axis_index] < min_distance:
                continue
            lane_kept.append(rect)
        if allow_gap_fill and len(lane_kept) >= 3:
            axis_size = float(np.median([rect[axis_size_index] for rect in lane_kept]))
            diffs = np.diff([rect[axis_index] for rect in lane_kept])
            sane_diffs = [diff for diff in diffs if axis_size * 0.75 <= diff <= axis_size * 1.55]
            typical_pitch = float(np.median(sane_diffs)) if sane_diffs else float(np.median(diffs))
            filled: list[tuple[int, int, int, int]] = []
            for current, following in zip(lane_kept, lane_kept[1:]):
                filled.append(current)
                gap = following[axis_index] - current[axis_index]
                if typical_pitch > axis_size * 0.75 and gap > typical_pitch * 1.55:
                    missing_count = min(2, max(1, int(round(gap / typical_pitch)) - 1))
                    for missing_index in range(1, missing_count + 1):
                        new_axis = int(round(current[axis_index] + gap * missing_index / (missing_count + 1)))
                        if orientation == "vertical":
                            filled.append((current[0], new_axis, current[2], current[3]))
                        else:
                            filled.append((new_axis, current[1], current[2], current[3]))
            filled.append(lane_kept[-1])
            lane_kept = filled
        kept.extend(lane_kept)

    if orientation == "vertical":
        kept.sort(key=lambda rect: (rect[1], rect[0]))
    else:
        kept.sort(key=lambda rect: (rect[0], rect[1]))
    return kept


def _row_content_score(opening: np.ndarray) -> np.ndarray:
    norm = cv2.equalizeHist(opening)
    edges = cv2.Canny(norm, 35, 110)
    edge_profile = edges.mean(axis=1) / 255.0
    non_dark_profile = (opening > 70).mean(axis=1)
    texture_profile = np.clip(opening.std(axis=1) / 80.0, 0, 1)
    return np.maximum.reduce([edge_profile * 2.0, non_dark_profile, texture_profile])


def _align_rects_to_photo_content(
    gray: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    orientation: str,
) -> list[tuple[int, int, int, int]]:
    if not rects:
        return rects

    aligned: list[tuple[int, int, int, int]] = []
    for x, y, w, h in rects:
        if orientation == "vertical":
            search_margin = max(4, int(round(h * 0.18)))
            y_min = max(0, y - search_margin)
            y_max = min(gray.shape[0] - h, y + search_margin)
            opening = gray[y_min : y_max + h, x : x + w]
            score = _row_content_score(opening)
            window = np.ones(h, dtype=np.float32)
            sums = np.convolve(score, window, mode="valid")
            if len(sums) == 0:
                aligned.append((x, y, w, h))
                continue
            # Penalize black-heavy crop edges so the window slides away from separators.
            dark = opening <= 70
            top_edge = np.convolve(dark.mean(axis=1), np.ones(max(2, h // 18)), mode="valid")
            edge_penalty = np.zeros_like(sums)
            edge_window = max(2, h // 18)
            for idx in range(len(sums)):
                top = dark[idx : idx + edge_window].mean()
                bottom = dark[idx + h - edge_window : idx + h].mean()
                edge_penalty[idx] = top + bottom
            best = int(np.argmax(sums - edge_penalty * h * 0.25))
            aligned.append((x, y_min + best, w, h))
        else:
            search_margin = max(4, int(round(w * 0.18)))
            x_min = max(0, x - search_margin)
            x_max = min(gray.shape[1] - w, x + search_margin)
            opening = gray[y : y + h, x_min : x_max + w].T
            score = _row_content_score(opening)
            window = np.ones(w, dtype=np.float32)
            sums = np.convolve(score, window, mode="valid")
            if len(sums) == 0:
                aligned.append((x, y, w, h))
                continue
            dark = opening <= 70
            edge_penalty = np.zeros_like(sums)
            edge_window = max(2, w // 18)
            for idx in range(len(sums)):
                left = dark[idx : idx + edge_window].mean()
                right = dark[idx + w - edge_window : idx + w].mean()
                edge_penalty[idx] = left + right
            best = int(np.argmax(sums - edge_penalty * w * 0.25))
            aligned.append((x_min + best, y, w, h))

    return aligned


def _regular_grid_candidates(
    gray: np.ndarray,
    orientation_override: Optional[str] = None,
    base_sensitivity: int = 18,
    lane_count: int = 1,
    aspect_mode: str = "Landscape 3:2",
    expected_frames_per_lane: int = 0,
) -> tuple[list[tuple[int, int, int, int]], str]:
    x, y, w, h = _strip_bounds(gray)
    crop = gray[y : y + h, x : x + w]
    orientation = orientation_override or ("horizontal" if w >= h else "vertical")

    border_size = max(1, min(crop.shape) // 20)
    border = np.concatenate(
        [
            crop[:border_size, :].ravel(),
            crop[-border_size:, :].ravel(),
            crop[:, :border_size].ravel(),
            crop[:, -border_size:].ravel(),
        ]
    )
    background = _dominant_gray_value(border)
    rects: list[tuple[int, int, int, int]] = []
    landscape = aspect_mode != "Portrait 2:3"

    lanes = _lanes_for_crop(crop, orientation, lane_count)

    for lane_start, lane_end in lanes:
        if lane_end <= lane_start:
            continue

        if orientation == "horizontal":
            lane_crop = crop[lane_start:lane_end, :]
            lane_mask = cv2.absdiff(lane_crop, np.full_like(lane_crop, background)) >= base_sensitivity
            cross_profile = lane_mask.mean(axis=1)
            cross_start, cross_end = _largest_active_span(cross_profile, 0.12, lane_end - lane_start)
            content = lane_crop[cross_start:cross_end, :]
            axis_len = w
            cross_len = max(1, cross_end - cross_start)
            frame_axis = int(round(cross_len * 1.5 if landscape else cross_len / 1.5))
            axis_profile = (
                cv2.absdiff(content, np.full_like(content, background)) >= base_sensitivity
            ).mean(axis=0)
        else:
            lane_crop = crop[:, lane_start:lane_end]
            lane_mask = cv2.absdiff(lane_crop, np.full_like(lane_crop, background)) >= base_sensitivity
            cross_profile = lane_mask.mean(axis=0)
            cross_start, cross_end = _largest_active_span(cross_profile, 0.12, lane_end - lane_start)
            content = lane_crop[:, cross_start:cross_end]
            axis_len = h
            cross_len = max(1, cross_end - cross_start)
            frame_axis = int(round(cross_len / 1.5 if landscape else cross_len * 1.5))
            axis_profile = (
                cv2.absdiff(content, np.full_like(content, background)) >= base_sensitivity
            ).mean(axis=1)

        frame_axis = max(8, frame_axis)
        active = axis_profile > 0.18
        active = _close_small_gaps(active, max(2, int(frame_axis * 0.12)))
        segments = _segments_from_active(active, min_length=max(8, int(frame_axis * 0.28)))
        centers = [float(start + end) / 2 for start, end in segments]
        centers = _regularize_centers(centers, frame_axis, axis_len, expected_frames_per_lane)

        for center in centers:
            axis_start = int(round(center - frame_axis / 2))
            axis_end = axis_start + frame_axis
            if axis_end <= 0 or axis_start >= axis_len:
                continue
            axis_start = max(0, axis_start)
            axis_end = min(axis_len, axis_end)

            if orientation == "horizontal":
                rects.append(
                    (
                        x + axis_start,
                        y + lane_start + cross_start,
                        axis_end - axis_start,
                        cross_len,
                    )
                )
            else:
                rects.append(
                    (
                        x + lane_start + cross_start,
                        y + axis_start,
                        cross_len,
                        axis_end - axis_start,
                    )
                )

    if len(rects) < 1:
        return [], orientation
    return _normalize_rects_to_common_aspect(rects, gray.shape[1], gray.shape[0], orientation, aspect_mode), orientation


def _edge_positions_from_dark_bands(
    profile: np.ndarray,
    min_distance: int,
    expected_count: int,
) -> list[float]:
    smooth_size = max(5, int(min_distance * 0.15) | 1)
    smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_size, 1), 0).ravel()
    threshold = max(0.38, float(np.percentile(smoothed, 82)))
    active = smoothed > threshold
    active = _close_small_gaps(active, max(2, int(min_distance * 0.12)))
    band_segments = _segments_from_active(active, min_length=max(2, int(min_distance * 0.06)))
    centers = [float(start + end) / 2 for start, end in band_segments]
    centers = sorted(centers)

    if expected_count > 0 and len(centers) >= 2:
        usable = centers
        if len(usable) > expected_count + 1:
            scores = [smoothed[int(round(center))] for center in usable]
            ranked = sorted(zip(scores, usable), reverse=True)[: expected_count + 1]
            usable = sorted(center for _, center in ranked)
        if len(usable) >= 2:
            return np.linspace(usable[0], usable[-1], expected_count + 1).tolist()
    return centers


def _photo_opening_span(lane_crop: np.ndarray, orientation: str, black_threshold: int) -> tuple[int, int]:
    dark = lane_crop <= black_threshold
    if orientation == "horizontal":
        dark_profile = dark.mean(axis=0)
        full_start, full_end = _largest_active_span(dark_profile, 0.05, lane_crop.shape[1])
        inside = lane_crop[:, full_start:full_end]
        if inside.size == 0:
            return (0, lane_crop.shape[1])
        non_black_profile = (inside > black_threshold).mean(axis=0)
    else:
        dark_profile = dark.mean(axis=0)
        full_start, full_end = _largest_active_span(dark_profile, 0.05, lane_crop.shape[1])
        inside = lane_crop[:, full_start:full_end]
        if inside.size == 0:
            return (0, lane_crop.shape[1])
        non_black_profile = (inside > black_threshold).mean(axis=0)

    threshold = max(0.10, min(0.45, float(non_black_profile.mean() * 0.45)))
    inner_start, inner_end = _largest_active_span(non_black_profile, threshold, max(1, full_end - full_start))
    if inner_end - inner_start < max(8, (full_end - full_start) * 0.35):
        return (full_start, full_end)
    return (full_start + inner_start, full_start + inner_end)


def _content_segments_from_opening(opening: np.ndarray, frame_axis: int, expected_count: int = 0) -> list[tuple[int, int]]:
    norm = cv2.equalizeHist(opening)
    edges = cv2.Canny(norm, 35, 110)
    edge_profile = edges.mean(axis=1) / 255.0
    non_dark_profile = (opening > 70).mean(axis=1)
    texture_profile = np.clip(opening.std(axis=1) / 80.0, 0, 1)
    score = np.maximum.reduce([edge_profile * 2.0, non_dark_profile, texture_profile])

    min_length = max(35, int(frame_axis * 0.22))
    max_gap = max(3, int(frame_axis * 0.025))
    best_segments: list[tuple[int, int]] = []
    best_key = (-9999, 0)

    for percentile in (8, 10, 12, 15, 18, 20, 22, 25):
        threshold = max(0.10, float(np.percentile(score, percentile)))
        active = score > threshold
        active = _close_small_gaps(active, max_gap)
        segments = _segments_from_active(active, min_length=min_length)
        if not segments:
            continue
        large_count = sum(1 for start, end in segments if end - start > frame_axis * 1.45)
        tiny_count = sum(1 for start, end in segments if end - start < frame_axis * 0.35)
        key = (len(segments) - large_count * 3 - tiny_count, len(segments))
        if key > best_key:
            best_key = key
            best_segments = segments

    if expected_count > 0 and best_segments:
        centers = [float(start + end) / 2 for start, end in best_segments]
        centers = _regularize_centers(centers, frame_axis, opening.shape[0], expected_count)
        return [
            (
                max(0, int(round(center - frame_axis / 2))),
                min(opening.shape[0], int(round(center + frame_axis / 2))),
            )
            for center in centers
        ]

    if len(best_segments) >= 2:
        starts = [start for start, _ in best_segments]
        centers = [float(start + end) / 2 for start, end in best_segments]
        diffs = np.diff(centers)
        sane_diffs = [diff for diff in diffs if frame_axis * 0.55 <= diff <= frame_axis * 1.8]
        pitch = float(np.median(sane_diffs)) if sane_diffs else float(np.median(diffs))
        if best_segments[0][0] > frame_axis * 0.55 and pitch > frame_axis * 0.5:
            center = centers[0] - pitch
            if center + frame_axis / 2 > 0:
                best_segments.insert(0, (max(0, int(round(center - frame_axis / 2))), int(round(center + frame_axis / 2))))
        if opening.shape[0] - best_segments[-1][1] > frame_axis * 0.55 and pitch > frame_axis * 0.5:
            center = centers[-1] + pitch
            if center - frame_axis / 2 < opening.shape[0]:
                best_segments.append((int(round(center - frame_axis / 2)), min(opening.shape[0], int(round(center + frame_axis / 2)))))

    return best_segments


def _black_border_pitch_candidates(
    gray: np.ndarray,
    orientation_override: Optional[str] = None,
    lane_count: int = 1,
    aspect_mode: str = "Landscape 3:2",
    expected_frames_per_lane: int = 0,
    black_threshold: int = 70,
) -> tuple[list[tuple[int, int, int, int]], str]:
    x, y, w, h = _strip_bounds(gray)
    crop = gray[y : y + h, x : x + w]
    orientation = orientation_override or ("horizontal" if w >= h else "vertical")
    landscape = aspect_mode != "Portrait 2:3"

    lanes = _lanes_for_crop(crop, orientation, lane_count)

    rects: list[tuple[int, int, int, int]] = []
    for lane_start, lane_end in lanes:
        if lane_end <= lane_start:
            continue

        if orientation == "horizontal":
            lane_crop = crop[lane_start:lane_end, :]
            cross_start, cross_end = _photo_opening_span(lane_crop.T, "horizontal", black_threshold)
            cross_len = max(1, cross_end - cross_start)
            frame_axis = int(round(cross_len * 1.5 if landscape else cross_len / 1.5))
            dark = lane_crop <= black_threshold
            band_profile = dark[cross_start:cross_end, :].mean(axis=0)
            axis_len = w
        else:
            lane_crop = crop[:, lane_start:lane_end]
            cross_start, cross_end = _photo_opening_span(lane_crop, "vertical", black_threshold)
            cross_len = max(1, cross_end - cross_start)
            frame_axis = int(round(cross_len / 1.5 if landscape else cross_len * 1.5))
            dark = lane_crop <= black_threshold
            band_profile = dark[:, cross_start:cross_end].mean(axis=1)
            axis_len = h

        frame_axis = max(8, frame_axis)
        starts: list[float] = []
        if orientation == "horizontal":
            opening = lane_crop[cross_start:cross_end, :].T
        else:
            opening = lane_crop[:, cross_start:cross_end]
        photo_segments = _content_segments_from_opening(opening, frame_axis, expected_frames_per_lane)
        if photo_segments:
            if expected_frames_per_lane > 0:
                starts = [float(start + end) / 2 - frame_axis / 2 for start, end in photo_segments]
            else:
                starts = [float(start) for start, _ in photo_segments]

        edges = _edge_positions_from_dark_bands(band_profile, frame_axis, expected_frames_per_lane)

        if not starts and expected_frames_per_lane > 0 and len(edges) >= 2:
            for index in range(expected_frames_per_lane):
                if index + 1 >= len(edges):
                    break
                center = (edges[index] + edges[index + 1]) / 2
                starts.append(center - frame_axis / 2)
        elif not starts and len(edges) >= 2:
            diffs = np.diff(edges)
            pitch = float(np.median(diffs))
            if pitch > frame_axis * 0.45:
                first_center = (edges[0] + edges[1]) / 2
                last_center = (edges[-2] + edges[-1]) / 2
                count = max(1, int(round((last_center - first_center) / pitch)) + 1)
                for center in np.linspace(first_center, last_center, count):
                    starts.append(float(center) - frame_axis / 2)
        elif not starts:
            # Fallback: use visible non-white/dark activity, then regularize it.
            active = band_profile > max(0.12, float(np.percentile(band_profile, 55)))
            active = _close_small_gaps(active, max(2, int(frame_axis * 0.15)))
            segments = _segments_from_active(active, min_length=max(8, int(frame_axis * 0.3)))
            centers = [float(start + end) / 2 for start, end in segments]
            centers = _regularize_centers(centers, frame_axis, axis_len, expected_frames_per_lane)
            starts = [center - frame_axis / 2 for center in centers]

        for start in starts:
            axis_start = int(round(start))
            axis_end = axis_start + frame_axis
            if axis_end <= 0 or axis_start >= axis_len:
                continue
            axis_start = max(0, axis_start)
            axis_end = min(axis_len, axis_end)

            if orientation == "horizontal":
                rects.append(
                    (
                        x + axis_start,
                        y + lane_start + cross_start,
                        axis_end - axis_start,
                        cross_len,
                    )
                )
            else:
                rects.append(
                    (
                        x + lane_start + cross_start,
                        y + axis_start,
                        cross_len,
                        axis_end - axis_start,
                    )
                )

    if not rects:
        return [], orientation
    rects = _normalize_rects_to_common_aspect(rects, gray.shape[1], gray.shape[0], orientation, aspect_mode)
    return _dedupe_same_lane_rects(rects, orientation), orientation


def _frame_band_candidates(
    gray: np.ndarray,
    orientation_override: Optional[str] = None,
    base_sensitivity: int = 18,
    lane_count: int = 1,
) -> tuple[list[tuple[int, int, int, int]], str]:
    height, width = gray.shape
    x, y, w, h = _strip_bounds(gray)
    crop = gray[y : y + h, x : x + w]
    orientation = orientation_override or ("horizontal" if w >= h else "vertical")

    border_size = max(1, min(crop.shape) // 20)
    border = np.concatenate(
        [
            crop[:border_size, :].ravel(),
            crop[-border_size:, :].ravel(),
            crop[:, :border_size].ravel(),
            crop[:, -border_size:].ravel(),
        ]
    )
    background = _dominant_gray_value(border)
    rects: list[tuple[int, int, int, int]] = []

    lanes = _lanes_for_crop(crop, orientation, lane_count)

    for lane_start, lane_end in lanes:
        if lane_end <= lane_start:
            continue

        if orientation == "horizontal":
            lane_crop = crop[lane_start:lane_end, :]
            axis_len = w
            cross_len = lane_end - lane_start
            lane_offset = lane_start
            base_profile = (
                cv2.absdiff(lane_crop, np.full_like(lane_crop, background)) >= base_sensitivity
            ).mean(axis=0)
        else:
            lane_crop = crop[:, lane_start:lane_end]
            axis_len = h
            cross_len = lane_end - lane_start
            lane_offset = lane_start
            base_profile = (
                cv2.absdiff(lane_crop, np.full_like(lane_crop, background)) >= base_sensitivity
            ).mean(axis=1)

        active = base_profile > 0.22
        active = _close_small_gaps(active, max(2, int(axis_len * 0.015)))
        segments = _segments_from_active(active, min_length=max(20, int(axis_len * 0.035)))

        if len(segments) < 2:
            norm = normalize_gray(lane_crop)
            edges = cv2.Canny(norm, 35, 120)
            lane_contrast = cv2.absdiff(lane_crop, np.full_like(lane_crop, background))
            if orientation == "horizontal":
                edge_profile = edges.mean(axis=0)
                contrast_profile = lane_contrast.mean(axis=0)
            else:
                edge_profile = edges.mean(axis=1)
                contrast_profile = lane_contrast.mean(axis=1)

            profile = edge_profile + 0.45 * contrast_profile
            smooth_size = max(9, int(axis_len * 0.012) | 1)
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_size, 1), 0).ravel()
            threshold = max(float(np.percentile(profile, 45)), float(profile.mean() * 0.72))
            active = profile > threshold
            active = _close_small_gaps(active, max(2, int(axis_len * 0.012)))
            segments = _segments_from_active(active, min_length=max(20, int(axis_len * 0.045)))

        margin = max(2, int(cross_len * 0.035))
        if len(segments) < 1:
            continue

        for start, end in segments:
            if end <= start:
                continue
            if orientation == "horizontal":
                rects.append((x + start, y + lane_offset + margin, end - start, max(1, cross_len - 2 * margin)))
            else:
                rects.append((x + lane_offset + margin, y + start, max(1, cross_len - 2 * margin), end - start))

    if len(rects) < 2:
        return [], orientation

    return rects, orientation


def detect_frames(
    preview: Image.Image,
    preview_scale: float,
    min_frame_area_percent: float = 0.8,
    padding_percent: float = 0.0,
    detection_method: str = "Frame bands",
    orientation_override: Optional[str] = None,
    base_sensitivity: int = 18,
    lane_count: int = 1,
    aspect_mode: str = "Landscape 3:2",
    expected_frames_per_lane: int = 0,
    black_threshold: int = 70,
) -> tuple[list[CropBox], str]:
    gray = pil_to_cv_gray(preview)
    height, width = gray.shape

    requested_orientation = None
    if orientation_override in {"horizontal", "vertical"}:
        requested_orientation = orientation_override

    candidates: list[tuple[int, int, int, int]] = []
    orientation = requested_orientation or ("horizontal" if width >= height else "vertical")

    if detection_method == "Black border pitch":
        candidates, orientation = _black_border_pitch_candidates(
            gray,
            requested_orientation,
            lane_count,
            aspect_mode,
            expected_frames_per_lane,
            black_threshold,
        )

    if detection_method == "Regular 3:2 grid":
        candidates, orientation = _regular_grid_candidates(
            gray,
            requested_orientation,
            base_sensitivity,
            lane_count,
            aspect_mode,
            expected_frames_per_lane,
        )

    if detection_method in {"Frame bands", "Auto"}:
        candidates, orientation = _frame_band_candidates(gray, requested_orientation, base_sensitivity, lane_count)

    if detection_method == "Contours" or (detection_method == "Auto" and len(candidates) < 2):
        candidates = _contour_frame_candidates(gray)
        min_area = width * height * (min_frame_area_percent / 100.0)
        candidates = [rect for rect in candidates if rect[2] * rect[3] >= min_area]
        orientation = requested_orientation or detect_orientation(candidates, gray.shape)

    if len(candidates) < 2:
        candidates = _projection_fallback(gray)
        orientation = requested_orientation or detect_orientation(candidates, gray.shape)

    if orientation == "horizontal":
        candidates.sort(key=lambda r: (r[0], r[1]))
    else:
        candidates.sort(key=lambda r: (r[1], r[0]))

    pad_factor = padding_percent / 100.0
    boxes: list[CropBox] = []
    inv = 1.0 / max(preview_scale, 1e-9)
    for index, (x, y, w, h) in enumerate(candidates, start=1):
        pad_x = w * pad_factor
        pad_y = h * pad_factor
        ox = int(round((x - pad_x) * inv))
        oy = int(round((y - pad_y) * inv))
        ow = int(round((w + 2 * pad_x) * inv))
        oh = int(round((h + 2 * pad_y) * inv))
        if aspect_mode == "Landscape 3:2":
            center_x = ox + ow / 2
            center_y = oy + oh / 2
            ow = max(3, int(round(ow / 3)) * 3)
            oh = ow * 2 // 3
            ox = int(round(center_x - ow / 2))
            oy = int(round(center_y - oh / 2))
        elif aspect_mode == "Portrait 2:3":
            center_x = ox + ow / 2
            center_y = oy + oh / 2
            ow = max(2, int(round(ow / 2)) * 2)
            oh = ow * 3 // 2
            ox = int(round(center_x - ow / 2))
            oy = int(round(center_y - oh / 2))
        boxes.append(CropBox(ox, oy, ow, oh, index))

    return boxes, orientation


def draw_boxes_on_preview(
    preview: Image.Image,
    boxes: Iterable[CropBox],
    preview_scale: float,
) -> Image.Image:
    display = preview.convert("RGB").copy()
    arr = np.array(display)

    for box in boxes:
        if not box.enabled:
            color = (150, 150, 150)
        else:
            color = (255, 48, 48)
        x1 = int(round(box.x * preview_scale))
        y1 = int(round(box.y * preview_scale))
        x2 = int(round((box.x + box.width) * preview_scale))
        y2 = int(round((box.y + box.height) * preview_scale))
        thickness = max(1, int(round(min(arr.shape[:2]) / 900)))
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, thickness)
        label = str(box.order)
        label_width = max(28, 14 + 14 * len(label))
        label_height = 22
        label_y1 = max(0, y1 - label_height - 2)
        label_y2 = label_y1 + label_height
        cv2.rectangle(arr, (x1, label_y1), (x1 + label_width, label_y2), color, -1)
        cv2.putText(
            arr,
            label,
            (x1 + 5, label_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )

    return Image.fromarray(arr)


def trim_border(image: Image.Image, tolerance: int = 12) -> Image.Image:
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    diff = ImageChops.difference(rgb, bg)
    diff = ImageOps.grayscale(diff)
    mask = diff.point(lambda p: 255 if p > tolerance else 0)
    bbox = mask.getbbox()
    return image.crop(bbox) if bbox else image


def invert_negative(image: Image.Image, normalize_colors: bool = False) -> Image.Image:
    if image.mode == "RGBA":
        rgb, alpha = image.convert("RGB"), image.getchannel("A")
        inverted = ImageOps.invert(rgb)
        if normalize_colors:
            inverted = ImageOps.autocontrast(inverted, cutoff=0.5)
        inverted.putalpha(alpha)
        return inverted

    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    inverted = ImageOps.invert(image)
    if normalize_colors:
        inverted = ImageOps.autocontrast(inverted, cutoff=0.5)
    return inverted


def _metadata_for_save(original: Image.Image, fmt: str) -> dict:
    info: dict = {}
    if "dpi" in original.info:
        info["dpi"] = original.info["dpi"]
    if "icc_profile" in original.info:
        info["icc_profile"] = original.info["icc_profile"]
    if fmt == "JPEG" and "exif" in original.info:
        info["exif"] = original.info["exif"]
    return info


def save_image_bytes(image: Image.Image, fmt: str, original: Image.Image) -> bytes:
    output = io.BytesIO()
    metadata = _metadata_for_save(original, fmt)

    if fmt == "JPEG":
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(
            output,
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
            progressive=False,
            **metadata,
        )
    elif fmt == "PNG":
        image.save(output, format="PNG", compress_level=0, **metadata)
    elif fmt == "TIFF":
        image.save(output, format="TIFF", compression="tiff_lzw", **metadata)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return output.getvalue()


def export_zip(
    original: Image.Image,
    boxes: Iterable[CropBox],
    fmt: str = "TIFF",
    trim_borders: bool = False,
    invert: bool = False,
    normalize_negative: bool = False,
) -> bytes:
    extension = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif"}[fmt]
    selected = [box.clamp(*original.size) for box in boxes if box.enabled]
    selected.sort(key=lambda box: box.order)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for export_index, box in enumerate(selected, start=1):
            crop = original.crop((box.x, box.y, box.right, box.bottom))
            if trim_borders:
                crop = trim_border(crop)
            if invert:
                crop = invert_negative(crop, normalize_colors=normalize_negative)
            frame_bytes = save_image_bytes(crop, fmt, original)
            archive.writestr(f"frame_{export_index:03d}.{extension}", frame_bytes)
    return zip_buffer.getvalue()


def export_zip_file(
    original: Image.Image,
    boxes: Iterable[CropBox],
    output_path: Path,
    fmt: str = "TIFF",
    trim_borders: bool = False,
    invert: bool = False,
    normalize_negative: bool = False,
) -> Path:
    """Write crops to a ZIP on disk, which is safer for very large scans."""
    extension = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif"}[fmt]
    selected = [box.clamp(*original.size) for box in boxes if box.enabled]
    selected.sort(key=lambda box: box.order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for export_index, box in enumerate(selected, start=1):
            crop = original.crop((box.x, box.y, box.right, box.bottom))
            if trim_borders:
                crop = trim_border(crop)
            if invert:
                crop = invert_negative(crop, normalize_colors=normalize_negative)
            frame_bytes = save_image_bytes(crop, fmt, original)
            archive.writestr(f"frame_{export_index:03d}.{extension}", frame_bytes)
            crop.close()
    return output_path


def estimate_global_skew_degrees(preview: Image.Image) -> float:
    gray = pil_to_cv_gray(preview)
    norm = normalize_gray(gray)
    edges = cv2.Canny(norm, 40, 130)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(30, int(min(gray.shape) * 0.2)),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -20 <= angle <= 20:
            angles.append(angle)
        elif 70 <= abs(angle) <= 110:
            angles.append(angle - 90 if angle > 0 else angle + 90)

    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_image_for_detection(image: Image.Image, angle_degrees: float) -> Image.Image:
    if abs(angle_degrees) < 0.05:
        return image
    return image.rotate(-angle_degrees, expand=True, resample=Image.Resampling.BICUBIC)
