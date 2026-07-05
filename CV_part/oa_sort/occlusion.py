from __future__ import annotations

import math

import numpy as np


_DEFAULT_GM_KX = 3.0 * math.sqrt(2.0)
_DEFAULT_GM_KY = 3.0
_EPS = 1e-6


def ensure_boxes_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Normalize boxes to a float32 (N, 4) xyxy array."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(f"Expected boxes with at least 4 columns, got shape {arr.shape}")
    return arr[:, :4].astype(np.float32, copy=False)


def box_areas_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = ensure_boxes_xyxy(boxes)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return widths * heights


def intersection_area_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = ensure_boxes_xyxy(boxes_a)
    boxes_b = ensure_boxes_xyxy(boxes_b)

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    widths = np.maximum(0.0, right - left)
    heights = np.maximum(0.0, bottom - top)
    return (widths * heights).astype(np.float32, copy=False)


def pairwise_iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = ensure_boxes_xyxy(boxes_a)
    boxes_b = ensure_boxes_xyxy(boxes_b)

    inter = intersection_area_matrix(boxes_a, boxes_b)
    if inter.size == 0:
        return inter

    area_a = box_areas_xyxy(boxes_a)[:, None]
    area_b = box_areas_xyxy(boxes_b)[None, :]
    union = np.maximum(area_a + area_b - inter, _EPS)
    return (inter / union).astype(np.float32, copy=False)


def build_occlusion_relationships(boxes: np.ndarray, occ_trigger_thresh: float = 5.0) -> np.ndarray:
    """Return a mask where mask[i, j] means box j occludes box i."""
    boxes = ensure_boxes_xyxy(boxes)
    if len(boxes) == 0:
        return np.zeros((0, 0), dtype=bool)

    inter = intersection_area_matrix(boxes, boxes)
    np.fill_diagonal(inter, 0.0)

    bottoms = boxes[:, 3]
    bottom_gap = bottoms[None, :] - bottoms[:, None]
    return (inter > 0.0) & (bottom_gap >= float(occ_trigger_thresh))


def compute_occlusion_coefficients(
    boxes: np.ndarray,
    occ_trigger_thresh: float = 5.0,
    gm_kx: float = _DEFAULT_GM_KX,
    gm_ky: float = _DEFAULT_GM_KY,
) -> np.ndarray:
    """Compute refined OA-SORT occlusion coefficients for a set of boxes."""
    boxes = ensure_boxes_xyxy(boxes)
    num_boxes = len(boxes)
    if num_boxes == 0:
        return np.empty((0,), dtype=np.float32)

    relationships = build_occlusion_relationships(boxes, occ_trigger_thresh=occ_trigger_thresh)
    inter = intersection_area_matrix(boxes, boxes)
    areas = np.maximum(box_areas_xyxy(boxes), _EPS)
    coeffs = np.zeros((num_boxes,), dtype=np.float32)

    if not relationships.any():
        return coeffs

    overlap_candidates = inter > 0.0
    np.fill_diagonal(overlap_candidates, True)

    int_boxes = np.column_stack(
        [
            np.floor(boxes[:, 0]),
            np.floor(boxes[:, 1]),
            np.ceil(boxes[:, 2]),
            np.ceil(boxes[:, 3]),
        ]
    ).astype(np.int32, copy=False)

    for i in range(num_boxes):
        occluders = np.flatnonzero(relationships[i])
        if len(occluders) == 0:
            continue

        left_i, top_i, right_i, bottom_i = int_boxes[i]
        width_i = max(0, int(right_i - left_i))
        height_i = max(0, int(bottom_i - top_i))
        if width_i == 0 or height_i == 0:
            continue

        local_gm = np.zeros((height_i, width_i), dtype=np.float32)
        occlusion_mask = np.zeros((height_i, width_i), dtype=bool)

        # Only boxes overlapping the target box can affect the local cropped GM.
        candidates = np.flatnonzero(overlap_candidates[i])
        for n in candidates:
            x1_n, y1_n, x2_n, y2_n = boxes[n]
            left_n, top_n, right_n, bottom_n = int_boxes[n]

            patch_left = max(left_i, left_n)
            patch_top = max(top_i, top_n)
            patch_right = min(right_i, right_n)
            patch_bottom = min(bottom_i, bottom_n)
            if patch_right <= patch_left or patch_bottom <= patch_top:
                continue

            sigma_x = max((x2_n - x1_n) / gm_kx, _EPS)
            sigma_y = max((y2_n - y1_n) / gm_ky, _EPS)
            center_x = (x1_n + x2_n) / 2.0
            center_y = (y1_n + y2_n) / 2.0

            xs = np.arange(patch_left, patch_right, dtype=np.float32) + 0.5
            ys = np.arange(patch_top, patch_bottom, dtype=np.float32) + 0.5
            gx = np.exp(-((xs - center_x) ** 2) / (2.0 * sigma_x * sigma_x))
            gy = np.exp(-((ys - center_y) ** 2) / (2.0 * sigma_y * sigma_y))
            gaussian_patch = gy[:, None] * gx[None, :]

            local_slice_y = slice(patch_top - top_i, patch_bottom - top_i)
            local_slice_x = slice(patch_left - left_i, patch_right - left_i)
            current = local_gm[local_slice_y, local_slice_x]
            local_gm[local_slice_y, local_slice_x] = np.maximum(current, gaussian_patch)

        for j in occluders:
            patch_left = max(left_i, int_boxes[j, 0])
            patch_top = max(top_i, int_boxes[j, 1])
            patch_right = min(right_i, int_boxes[j, 2])
            patch_bottom = min(bottom_i, int_boxes[j, 3])
            if patch_right <= patch_left or patch_bottom <= patch_top:
                continue

            local_slice_y = slice(patch_top - top_i, patch_bottom - top_i)
            local_slice_x = slice(patch_left - left_i, patch_right - left_i)
            occlusion_mask[local_slice_y, local_slice_x] = True

        coeffs[i] = float(local_gm[occlusion_mask].sum() / areas[i])

    return np.clip(coeffs, 0.0, 1.0).astype(np.float32, copy=False)


def apply_occlusion_aware_offset(iou_matrix: np.ndarray, occlusion_coefficients: np.ndarray, tau: float) -> np.ndarray:
    """Compute OA-SORT's OAO similarity score matrix."""
    iou_matrix = np.asarray(iou_matrix, dtype=np.float32)
    occ = np.asarray(occlusion_coefficients, dtype=np.float32).reshape(1, -1)
    if iou_matrix.size == 0:
        return iou_matrix
    if iou_matrix.shape[1] != occ.shape[1]:
        raise ValueError(
            f"OAO expects one occlusion coefficient per track, got matrix {iou_matrix.shape} and coeffs {occ.shape}"
        )
    tau = float(tau)
    return tau * (1.0 - occ) + (1.0 - tau) * iou_matrix


def compute_bias_aware_momentum(iou_value: float | np.ndarray, last_occlusion_coefficient: float | np.ndarray) -> np.ndarray:
    """Compute OA-SORT's BAM scalar."""
    iou_value = np.asarray(iou_value, dtype=np.float32)
    last_occ = np.asarray(last_occlusion_coefficient, dtype=np.float32)
    return np.clip(iou_value * (1.0 - last_occ), 0.0, 1.0).astype(np.float32, copy=False)


def blend_boxes_xyxy(detection_box: np.ndarray, predicted_box: np.ndarray, bam: float) -> np.ndarray:
    """Blend detection and prediction boxes with BAM as the measurement trust."""
    detection_box = ensure_boxes_xyxy(detection_box)
    predicted_box = ensure_boxes_xyxy(predicted_box)
    if len(detection_box) != 1 or len(predicted_box) != 1:
        raise ValueError("blend_boxes_xyxy expects exactly one detection box and one predicted box")
    weight = float(np.clip(bam, 0.0, 1.0))
    return (weight * detection_box + (1.0 - weight) * predicted_box).astype(np.float32, copy=False)
