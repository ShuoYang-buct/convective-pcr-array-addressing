from typing import Any
import numpy as np
import cv2
SMALL_BOX_AREA_RATIO=.25
POSITION_OUTLIER_NN_FACTOR=2.
POSITION_OUTLIER_MIN_COUNT=6

def median_value(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)

def box_area(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

def contains_box(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

def overlaps_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    return inter_w > 0 and inter_h > 0

def suppress_position_outliers(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < POSITION_OUTLIER_MIN_COUNT:
        return candidates
    centers = [(float(item["center_x"]), float(item["center_y"])) for item in candidates]
    nearest_distances: list[float] = []
    for i, (x1, y1) in enumerate(centers):
        min_dist_sq: float | None = None
        for j, (x2, y2) in enumerate(centers):
            if i == j:
                continue
            dist_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            if min_dist_sq is None or dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        nearest_distances.append((min_dist_sq or 0.0) ** 0.5)
    median_nn = median_value(nearest_distances)
    if median_nn <= 0:
        return candidates
    threshold = median_nn * POSITION_OUTLIER_NN_FACTOR
    kept = [item for item, nn_dist in zip(candidates, nearest_distances) if nn_dist <= threshold]
    if not kept:
        return candidates
    return kept

def filter_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    areas = [float(item["area"]) for item in candidates]
    median_area = median_value(areas)
    min_area = median_area * SMALL_BOX_AREA_RATIO
    size_filtered = [item for item in candidates if float(item["area"]) >= min_area]
    size_filtered.sort(key=lambda item: (float(item["confidence"]), float(item["area"])), reverse=True)
    kept: list[dict[str, Any]] = []
    for item in size_filtered:
        item_box = item["xyxy"]
        skip = False
        for existing in kept:
            existing_box = existing["xyxy"]
            if (
                contains_box(existing_box, item_box)
                or contains_box(item_box, existing_box)
                or overlaps_box(existing_box, item_box)
            ):
                skip = True
                break
        if not skip:
            kept.append(item)
    return suppress_position_outliers(kept)

def normalize_box_sizes(candidates: list[dict[str, Any]], image_shape) -> list[dict[str, Any]]:
    if not candidates:
        return []
    widths = [float(item["width"]) for item in candidates]
    heights = [float(item["height"]) for item in candidates]
    target_w = median_value(widths)
    target_h = median_value(heights)
    if target_w <= 0 or target_h <= 0:
        return candidates
    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    half_w = target_w / 2.0
    half_h = target_h / 2.0
    normalized = []
    for item in candidates:
        cx = float(item["center_x"])
        cy = float(item["center_y"])
        x1 = cx - half_w
        x2 = cx + half_w
        y1 = cy - half_h
        y2 = cy + half_h
        if x1 < 0:
            shift = -x1
            x1 += shift
            x2 += shift
        if x2 > image_width:
            shift = x2 - image_width
            x1 -= shift
            x2 -= shift
        if y1 < 0:
            shift = -y1
            y1 += shift
            y2 += shift
        if y2 > image_height:
            shift = y2 - image_height
            y1 -= shift
            y2 -= shift
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(image_width), x2)
        y2 = min(float(image_height), y2)
        item["width"] = float(target_w)
        item["height"] = float(target_h)
        item["xyxy"] = (float(x1), float(y1), float(x2), float(y2))
        item["area"] = box_area(item["xyxy"])
        normalized.append(item)
    return normalized

def ensure_color_image(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        return np.zeros((16, 16, 3), dtype=np.uint8)
    if image.dtype == np.uint8:
        base = image
    else:
        base = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ensure_color_image(base)

def _to_candidate_from_xyxy(xyxy: tuple[float, float, float, float], confidence: float = 1.0, class_id: int | None = None, source: str = "manual"):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    item_xyxy = (x1, y1, x2, y2)
    return {
        "class_id": int(class_id) if class_id is not None else None,
        "confidence": float(confidence),
        "center_x": float(cx),
        "center_y": float(cy),
        "width": float(w),
        "height": float(h),
        "xyxy": item_xyxy,
        "area": box_area(item_xyxy),
        "source": source,
    }

_MODEL_CACHE={}
def predict_yolo_boxes(image, weights):
    from ultralytics import YOLO
    bgr=to_uint8_bgr(image)
    if str(weights) not in _MODEL_CACHE:_MODEL_CACHE[str(weights)]=YOLO(str(weights))
    model=_MODEL_CACHE[str(weights)]
    pred=model.predict(source=[bgr],conf=.25,verbose=False)[0]
    raw=[]
    if pred.boxes is not None:
        for box,confidence,class_id in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist()):
            raw.append(_to_candidate_from_xyxy(tuple(box),confidence,int(class_id),'yolo'))
    return raw

def prepare_candidates(raw, image_shape):
    # Copy each box because normalization updates widths, heights and xyxy in place.
    thresholded=[dict(item) for item in raw if item['confidence']>=.5]
    result=normalize_box_sizes(filter_candidates(thresholded),image_shape)
    return normalize_box_sizes(filter_candidates(result),image_shape)

def detect_candidate_stages(image, weights):
    raw=predict_yolo_boxes(image,weights)
    return {'raw':raw,'filtered':prepare_candidates(raw,image.shape)}

def detect_candidates(image, weights):
    return detect_candidate_stages(image,weights)['filtered']
