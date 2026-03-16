#!/usr/bin/env python3
# --------------------------------------------------------
# Plane Detection: Multi-Granularity Planar Region Detection
# via Normal-Cluster Dominance Test
# --------------------------------------------------------

import argparse
import builtins
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm.auto import tqdm
from torchvision import transforms

# from semantic_sam.BaseModel import BaseModel
# from semantic_sam.BaseModel import BaseModel
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from utils.sam_utils.amg import remove_small_regions
from tasks import automatic_mask_generator as sam_amg
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator

# ========================= 配置 =========================
PROJECT_ROOT = Path(__file__).parent
CKPT_PATH = PROJECT_ROOT / "ckpts" / "swinl_only_sam_many2many.pth"
CFG_PATH = "configs/semantic_sam_only_sa-1b_swinL.yaml"

# ROUNDS = [
#     {"name": "semantic", "levels": [1,2]},
#     {"name": "instance", "levels": [1, 2, 3]},
#     {"name": "part",     "levels": [1, 2, 3, 4, 5, 6]},
# ]

ROUNDS = [
    {"name": "semantic", "levels": [2]},
    {"name": "instance", "levels": [3]},
    {"name": "part",     "levels": [4,5,6]},
]
AREA_THRESHOLD = 0.7
CLUSTER = 5
MIN_PLANE_PIXELS = 2000
MIN_PLANE_AREA_RATIO = 0.005
NORMAL_MERGE_COLOR_THRESH = 50.0
NORMAL_EDGE_CANNY_LOW = 50
NORMAL_EDGE_CANNY_HIGH = 100
NORMAL_EDGE_DILATE = 3
FINAL_MASK_HOLE_AREA = 500
MAX_PLANE_MEAN_ANGLE = 13.0
MAX_PLANE_P95_ANGLE = 30.0
RANDOM_SEED = 22

PALETTE = [
    (230,  25,  75), (60,  180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


def log(message: str = "", quiet: bool = False):
    if not quiet:
        print(message)


def apply_quiet_mode(quiet: bool):
    if quiet:
        builtins.print = lambda *args, **kwargs: None


def progress(iterable, quiet: bool = False, **kwargs):
    return tqdm(
        iterable,
        disable=not quiet,
        dynamic_ncols=True,
        **kwargs,
    )


# ========================= 法向聚类工具函数 =========================

def seed_everything(seed: int = RANDOM_SEED):
    """
    固定 Python / NumPy / PyTorch 的随机性，尽量提高复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def kmeans_torch(x: torch.Tensor, num_clusters: int, num_iters: int = 10):
    """
    简单 K-Means（纯 PyTorch，CUDA 友好）。

    Args:
        x:            (N, D) float tensor，建议已在 CUDA 上
        num_clusters: 聚类中心数 K
        num_iters:    迭代次数

    Returns:
        labels  (N,)   每个点所属簇的索引
        centers (K, D) 聚类中心
    """
    device = x.device
    N, D = x.shape
    if N <= num_clusters:
        indices = torch.arange(N, device=device)
    else:
        indices = torch.linspace(0, N - 1, steps=num_clusters, device=device).round().long()
    centers = x[indices].clone()

    labels = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(num_iters):
        dist = torch.cdist(x, centers, p=2)   # (N, K)
        labels = dist.argmin(dim=1)            # (N,)
        for k in range(num_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = x[mask].mean(dim=0)

    return labels, centers


def merge_similar(masks: list, image_rgb: torch.Tensor, color_thresh: float = 50.0) -> list:
    """
    将颜色均值相近的法向聚类 mask 合并，减少过分割。

    Args:
        masks:       list of (H, W) uint8 Tensor
        image_rgb:   (H, W, 3) float Tensor，法向映射到 [0,255] 的图
        color_thresh: 均值颜色差 L2 距离阈值，小于此值则合并

    Returns:
        list of (H, W) uint8 Tensor
    """
    avg_colors = []
    for mask in masks:
        if mask.sum() > 0:
            avg_color = image_rgb[mask.bool()].mean(dim=0)
        else:
            avg_color = torch.zeros(3, device=image_rgb.device)
        avg_colors.append(avg_color)

    merged_masks = []
    used = torch.zeros(len(masks), dtype=torch.bool, device=image_rgb.device)

    for i in range(len(masks)):
        if used[i]:
            continue
        current_mask = masks[i].clone().bool()
        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            color_diff = torch.norm(avg_colors[i].float() - avg_colors[j].float())
            if color_diff < color_thresh:
                current_mask |= masks[j].bool()
                used[j] = True
        merged_masks.append(current_mask.to(torch.uint8))

    return merged_masks


def SplitPic(
    image_rgb: torch.Tensor,
    num_clusters: int = 5,
    merge_clusters: bool = True,
    color_thresh: float = NORMAL_MERGE_COLOR_THRESH,
) -> list:
    """
    对法向可视化图像做无监督颜色聚类，返回各聚类对应的空间掩码。

    Args:
        image_rgb:    (H, W, 3) float Tensor，法向映射到 [0,255] 的图
        num_clusters:   K-Means 初始簇数
        merge_clusters: 是否合并相近法向簇
        color_thresh:   法向簇合并阈值

    Returns:
        list of (H, W) uint8 Tensor，每个元素为一个聚类区域的掩码
    """
    device = image_rgb.device
    pixels = image_rgb.reshape(-1, 3).float()
    if device.type != 'cuda':
        pixels = pixels.cuda()
    labels, _ = kmeans_torch(pixels, num_clusters)
    labels = labels.to(image_rgb.device)

    cluster_map = labels.reshape(image_rgb.shape[:2]) + 1  # 1-indexed

    masks = []
    for cluster_idx in range(1, num_clusters + 1):
        mask = (cluster_map == cluster_idx).to(torch.uint8)
        if mask.sum() > 2000:
            masks.append(mask)

    if merge_clusters:
        masks = merge_similar(masks, image_rgb, color_thresh=color_thresh)
    return masks


def largest_connected_component(mask: np.ndarray, min_area: int = 1) -> np.ndarray | None:
    """
    返回 mask 的最大连通域。若最大连通域面积小于 min_area，则返回 None。
    """
    if not mask.any():
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask.astype(bool) if int(mask.sum()) >= min_area else None

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(component_areas.argmax()) + 1
    max_area = int(stats[max_idx, cv2.CC_STAT_AREA])
    if max_area < min_area:
        return None
    return (labels == max_idx)


def fill_mask_holes(mask: np.ndarray, area_thresh: int = FINAL_MASK_HOLE_AREA) -> np.ndarray:
    """
    对最终平面 mask 做小孔洞填补，避免 normal_edge 噪声带来的零碎空洞。
    """
    cleaned_mask, _ = remove_small_regions(mask.astype(bool), area_thresh, mode="holes")
    return cleaned_mask.astype(bool)


def normal_flatness(mask: np.ndarray, normal_tensor: torch.Tensor) -> dict:
    """
    统计 mask 内法向相对平均法向的离散度。
    """
    if not mask.any():
        return {
            "mean_angle": float("inf"),
            "p95_angle": float("inf"),
            "mean_normal_norm": 0.0,
        }

    mask_bool = torch.from_numpy(mask).to(device=normal_tensor.device, dtype=torch.bool)
    normals = normal_tensor[mask_bool]
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)

    mean_normal = normals.mean(dim=0)
    mean_normal_norm = float(torch.linalg.norm(mean_normal).item())
    if mean_normal_norm < 1e-8:
        return {
            "mean_angle": float("inf"),
            "p95_angle": float("inf"),
            "mean_normal_norm": 0.0,
        }

    mean_normal = mean_normal / mean_normal.norm().clamp(min=1e-8)
    dots = torch.clamp((normals * mean_normal).sum(dim=-1), -1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(dots))

    return {
        "mean_angle": float(angles.mean().item()),
        "p95_angle": float(torch.quantile(angles, 0.95).item()),
        "mean_normal_norm": mean_normal_norm,
    }


def compute_normal_edge_map(
    normal_map: np.ndarray,
    canny_low: int = NORMAL_EDGE_CANNY_LOW,
    canny_high: int = NORMAL_EDGE_CANNY_HIGH,
    dilate_kernel: int = NORMAL_EDGE_DILATE,
) -> np.ndarray:
    """
    从法线图提取 Sobel 梯度边缘，并做轻微膨胀，便于把相邻平面切开。
    """
    normal_float = normal_map.astype(np.float32)
    edge_maps = []
    for c in range(normal_float.shape[2]):
        channel = normal_float[:, :, c]
        grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_maps.append(gradient)

    combined_gradient = np.maximum.reduce(edge_maps)
    if combined_gradient.max() > 0:
        combined_gradient = (
            combined_gradient / combined_gradient.max() * 255.0
        ).astype(np.uint8)
    else:
        combined_gradient = combined_gradient.astype(np.uint8)

    edges = cv2.Canny(combined_gradient, canny_low, canny_high)
    if dilate_kernel > 1:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges > 0


def split_mask_by_normal_edges(
    mask: np.ndarray,
    normal_edge: np.ndarray,
    min_pixels: int,
) -> list:
    """
    用 normal_edge 从候选 mask 中剔除转角/边界像素，再按连通域拆分。
    """
    if not mask.any():
        return []

    split_mask = np.logical_and(mask, np.logical_not(normal_edge))
    if not split_mask.any():
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        split_mask.astype(np.uint8), connectivity=8
    )

    components = []
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        components.append(labels == comp_id)

    if components:
        return components

    largest = largest_connected_component(split_mask, min_area=min_pixels)
    return [largest] if largest is not None else []


def MaxOverlap(
    mask: np.ndarray,
    mask_list: list,
    area_threshold: float = 0.7,
) -> tuple:
    """
    找到与输入 mask 重叠最大的 1~2 个法向簇。
    当第一簇没有明显主导时，同时返回第二簇，便于从非平面物体内部切出平面块。
    """
    if not mask.any() or len(mask_list) == 0:
        return None, None, []

    mask_bool = torch.from_numpy(mask).cuda()
    overlaps = []
    for idx, cluster_mask in enumerate(mask_list):
        overlap_area = torch.logical_and(mask_bool, cluster_mask.bool()).sum().item()
        overlaps.append((int(overlap_area), idx))

    overlaps.sort(key=lambda x: x[0], reverse=True)
    raw_areas = [area for area, _ in overlaps]

    if overlaps[0][0] == 0:
        return None, None, raw_areas

    max_area, max_idx = overlaps[0]
    primary = (mask_list[max_idx], max_area)
    secondary = None

    if len(overlaps) > 1 and overlaps[1][0] > 0:
        second_area, second_idx = overlaps[1]
        ratio = (max_area - second_area) / float(max_area)
        if ratio <= area_threshold:
            secondary = (mask_list[second_idx], second_area)

    return primary, secondary, raw_areas


def extract_planar_submasks(
    effective_mask: np.ndarray,
    color_masks: list,
    normal_tensor: torch.Tensor,
    area_threshold: float = AREA_THRESHOLD,
    min_pixels: int = MIN_PLANE_PIXELS,
    max_mean_angle: float = MAX_PLANE_MEAN_ANGLE,
    max_p95_angle: float = MAX_PLANE_P95_ANGLE,
) -> tuple:
    """
    从一个较大的候选 mask 中切出 1~2 个更平坦的子平面。
    """
    if not effective_mask.any():
        return [], []

    mask_area = int(effective_mask.sum())
    primary, secondary, raw_areas = MaxOverlap(
        effective_mask, color_masks, area_threshold=area_threshold
    )
    if primary is None:
        return [], raw_areas

    source_mask = torch.from_numpy(effective_mask).cuda()
    planar_pieces = []
    for candidate in (primary, secondary):
        if candidate is None:
            continue

        cluster_mask, overlap_area = candidate
        submask = torch.logical_and(source_mask, cluster_mask.bool()).cpu().numpy()
        submask = largest_connected_component(submask, min_area=min_pixels)
        if submask is None:
            continue

        flatness = normal_flatness(submask, normal_tensor)
        if flatness["mean_angle"] > max_mean_angle:
            continue
        if flatness["p95_angle"] > max_p95_angle:
            continue

        planar_pieces.append({
            "mask": submask.astype(bool),
            "ratio": float(overlap_area) / max(mask_area, 1),
            "mean_angle": flatness["mean_angle"],
            "p95_angle": flatness["p95_angle"],
            "mean_normal_norm": flatness["mean_normal_norm"],
        })

    planar_pieces.sort(key=lambda x: x["ratio"], reverse=True)
    return planar_pieces, raw_areas


# ========================= 掩码生成 =========================

class FocusedSemanticSamAutomaticMaskGenerator(SemanticSamAutomaticMaskGenerator):
    """
    在不改动原始 Semantic-SAM 代码的前提下，支持用 focus_mask 约束采样点。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_focus_stats = self._make_focus_stats(enabled=False)

    @staticmethod
    def _make_focus_stats(enabled: bool) -> dict:
        return {
            "enabled": enabled,
            "total_points": 0,
            "kept_points": 0,
            "fallback_to_full_image": False,
        }

    @staticmethod
    def _empty_mask_data(device: torch.device) -> sam_amg.MaskData:
        return sam_amg.MaskData(
            rles=[],
            boxes=torch.empty((0, 4), dtype=torch.float32, device=device),
            iou_preds=torch.empty((0,), dtype=torch.float32, device=device),
            points=torch.empty((0, 4), dtype=torch.float32, device=device),
            stability_score=torch.empty((0,), dtype=torch.float32, device=device),
        )

    @staticmethod
    def _normalize_focus_mask(
        focus_mask: np.ndarray | torch.Tensor | None,
        expected_hw: tuple[int, int],
    ) -> np.ndarray | None:
        if focus_mask is None:
            return None
        if isinstance(focus_mask, torch.Tensor):
            focus_mask = focus_mask.detach().cpu().numpy()
        focus_mask = np.asarray(focus_mask)
        if focus_mask.ndim != 2:
            raise ValueError(f"focus_mask 必须是 2D，实际 shape={focus_mask.shape}")
        if tuple(focus_mask.shape) != tuple(expected_hw):
            raise ValueError(
                f"focus_mask 尺寸 {focus_mask.shape} 与输入图像尺寸 {expected_hw} 不一致"
            )
        return focus_mask.astype(bool, copy=False)

    @staticmethod
    def _resample_points_in_focus_mask(
        points_for_image: np.ndarray,
        crop_box: list[int],
        cropped_im_size: tuple[int, ...],
        focus_mask: np.ndarray | None,
        focus_stats: dict,
    ) -> np.ndarray:
        target_points = int(len(points_for_image))
        focus_stats["total_points"] += target_points
        if target_points == 0:
            return points_for_image

        if focus_mask is None:
            focus_stats["kept_points"] += target_points
            return points_for_image

        crop_h, crop_w = cropped_im_size
        x0, y0, _, _ = crop_box
        crop_focus = focus_mask[y0 : y0 + crop_h, x0 : x0 + crop_w]
        ys, xs = np.nonzero(crop_focus)
        available_points = int(len(xs))
        focus_stats["focus_pixels"] = focus_stats.get("focus_pixels", 0) + available_points
        if available_points == 0:
            return np.empty((0, 2), dtype=np.float32)

        replace = available_points < target_points
        sampled_idx = np.random.choice(
            available_points,
            size=target_points,
            replace=replace,
        )
        sampled_x = (xs[sampled_idx].astype(np.float32) + 0.5) / max(crop_w, 1)
        sampled_y = (ys[sampled_idx].astype(np.float32) + 0.5) / max(crop_h, 1)
        resampled_points = np.stack([sampled_x, sampled_y], axis=1).astype(np.float32)
        focus_stats["kept_points"] += int(len(resampled_points))
        focus_stats["resampled_with_replacement"] = (
            focus_stats.get("resampled_with_replacement", False) or replace
        )
        return resampled_points

    @torch.no_grad()
    def generate(
        self,
        image: np.ndarray,
        focus_mask: np.ndarray | torch.Tensor | None = None,
    ) -> list[dict]:
        focus_mask = self._normalize_focus_mask(focus_mask, image.shape[-2:])
        focus_stats = self._make_focus_stats(enabled=focus_mask is not None)
        mask_data = self._generate_masks(image, focus_mask=focus_mask, focus_stats=focus_stats)

        if (
            focus_mask is not None
            and focus_stats["total_points"] > 0
            and focus_stats["kept_points"] == 0
        ):
            focus_stats["fallback_to_full_image"] = True
            fallback_stats = self._make_focus_stats(enabled=False)
            mask_data = self._generate_masks(image, focus_mask=None, focus_stats=fallback_stats)
            focus_stats["fallback_total_points"] = fallback_stats["total_points"]
            focus_stats["fallback_kept_points"] = fallback_stats["kept_points"]

        self.last_focus_stats = focus_stats

        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                sam_amg.coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [
                sam_amg.rle_to_mask(rle) for rle in mask_data["rles"]
            ]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": sam_amg.area_from_rle(mask_data["rles"][idx]),
                "bbox": sam_amg.box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": sam_amg.box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(
        self,
        image: np.ndarray,
        focus_mask: np.ndarray | None = None,
        focus_stats: dict | None = None,
    ) -> sam_amg.MaskData:
        orig_size = image.shape[-2:]
        crop_boxes, layer_idxs = sam_amg.generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")
        data = self._empty_mask_data(device)

        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(
                image,
                crop_box,
                layer_idx,
                orig_size,
                focus_mask=focus_mask,
                focus_stats=focus_stats,
            )
            data.cat(crop_data)

        if len(crop_boxes) > 1 and len(data["boxes"]) > 0:
            scores = 1 / sam_amg.box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = sam_amg.batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros(len(data["boxes"]), device=data["boxes"].device),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: list[int],
        crop_layer_idx: int,
        orig_size: tuple[int, ...],
        focus_mask: np.ndarray | None = None,
        focus_stats: dict | None = None,
    ) -> sam_amg.MaskData:
        x0, y0, x1, y1 = crop_box
        cropped_im = image
        cropped_im_size = cropped_im.shape[-2:]
        device = image.device if isinstance(image, torch.Tensor) else torch.device("cpu")

        if focus_stats is None:
            focus_stats = self._make_focus_stats(enabled=focus_mask is not None)

        points_for_image = self.point_grids[crop_layer_idx]
        points_for_image = self._resample_points_in_focus_mask(
            points_for_image,
            crop_box,
            cropped_im_size,
            focus_mask,
            focus_stats,
        )

        if len(points_for_image) == 0:
            return self._empty_mask_data(device)

        data = self._empty_mask_data(device)
        self.enc_features = None
        for (points,) in sam_amg.batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(cropped_im, points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data

        if len(data["boxes"]) > 0:
            keep_by_nms = sam_amg.batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros(len(data["boxes"]), device=data["boxes"].device),
                iou_threshold=self.box_nms_thresh,
            )
            data.filter(keep_by_nms)

        data["boxes"] = sam_amg.uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["crop_boxes"] = torch.tensor(
            [crop_box for _ in range(len(data["rles"]))],
            dtype=torch.float32,
            device=device,
        )

        return data


def generate_masks_with_levels(
    model,
    image: Image.Image,
    levels: list,
    text_size: int = 640,
    focus_mask: np.ndarray | None = None,
) -> tuple[list, dict]:
    orig_w, orig_h = image.size
    t = [transforms.Resize(int(text_size), interpolation=Image.BICUBIC)]
    transform = transforms.Compose(t)
    image_transformed = transform(image)
    image_np = np.asarray(image_transformed)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

    focus_mask_resized = None
    if focus_mask is not None:
        focus_mask_resized = cv2.resize(
            focus_mask.astype(np.uint8),
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    mask_generator = FocusedSemanticSamAutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=10,
        level=levels,
    )
    outputs = mask_generator.generate(image_tensor, focus_mask=focus_mask_resized)
    focus_stats = dict(mask_generator.last_focus_stats)

    for mask_dict in outputs:
        seg = mask_dict['segmentation']
        resized = cv2.resize(
            seg.astype(np.uint8), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_dict['segmentation'] = resized.astype(bool)
    return outputs, focus_stats


# ========================= Debug 可视化 =========================

def _save_debug_round(
    round_idx: int,
    round_name: str,
    image_np: np.ndarray,
    remaining_before: np.ndarray,
    remaining_after: np.ndarray,
    all_masks: list,
    evaluated_records: list,   # list of (effective_mask, ratio, status)
                               # status: 'planar' | 'rejected' | 'skipped'
    cumulative_planes: list,   # ALL confirmed planes so far (across all rounds)
    color_masks_np: list,      # list of (H, W) bool np.ndarray，法向聚类区域
    normal_rgb_np: np.ndarray, # (H, W, 3) uint8，法向映射到 [0,255] 的图
    normal_edge_np: np.ndarray,# (H, W) bool，normal_rgb 上的边缘
    debug_dir: Path,
    quiet: bool = False,
):
    """
    保存单轮的全套 debug 图到 debug_dir/round_{idx}_{name}/

    输出文件：
        01_remaining_before.png  — 本轮开始前剩余区域（白=未处理，黑=已处理）
        02_all_masks_overlay.png — 本轮所有候选 mask（不同颜色）叠加原图
        03_planar_overlay.png    — 本轮确认平面（绿色）叠加原图
        04_rejected_overlay.png  — 本轮丢弃 mask（红色）叠加原图
        05_remaining_after.png   — 本轮结束后剩余区域
        06_summary_grid.png      — 上述 5 张并排总览
        07_{round_name}_planar.png — 本轮确认平面的纯彩色可视化
        08_cluster_map.png         — 本轮法向聚类区域可视化
        ../cluster_map.png         — debug 根目录共享的一份 cluster_map
        09_normal_edge.png         — 本轮 normal_rgb 的边缘图
        ../normal_edge.png         — debug 根目录共享的一份 normal_edge
        all_masks/                 — 本轮生成器输出的所有原始 mask
            mask_{j:03d}.png
        02b_all_masks_vis.png      — 所有原始 mask 的彩色汇总可视化
        evaluated/
            mask_{j:03d}_ratio{:.3f}_{PLANAR|REJECTED|SKIPPED}.png
    """
    h, w = image_np.shape[:2]
    rdir = debug_dir / f"round_{round_idx + 1:02d}_{round_name}"
    rdir.mkdir(parents=True, exist_ok=True)
    masks_dir = rdir / "evaluated"
    masks_dir.mkdir(exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── 保存所有原始 mask ──────────────────────────────────────────────
    all_masks_dir = rdir / "all_masks"
    all_masks_dir.mkdir(exist_ok=True)
    stale_all_masks_vis_dir = rdir / "all_masks_visualized"
    if stale_all_masks_vis_dir.exists():
        for stale_path in stale_all_masks_vis_dir.iterdir():
            if stale_path.is_file():
                stale_path.unlink()
        stale_all_masks_vis_dir.rmdir()

    all_masks_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for j, md in enumerate(all_masks):
        seg = md['segmentation']
        mask_img = (seg.astype(np.uint8) * 255)
        Image.fromarray(mask_img, mode='L').save(all_masks_dir / f"mask_{j:03d}.png")
        color = PALETTE[j % len(PALETTE)]
        for c in range(3):
            all_masks_vis[:, :, c][seg] = color[c]
    Image.fromarray(all_masks_vis).save(rdir / "02b_all_masks_vis.png")

    # ── 01 remaining before ──────────────────────────────────────────────
    rem_before_img = (remaining_before.astype(np.uint8) * 255)
    Image.fromarray(rem_before_img, mode='L').save(rdir / "01_remaining_before.png")

    # ── 02 all candidate masks (colored) overlaid on image ───────────────
    overlay_all = image_np.copy().astype(np.float64)
    for j, md in enumerate(all_masks):
        seg = md['segmentation']
        color = PALETTE[j % len(PALETTE)]
        for c in range(3):
            overlay_all[:, :, c][seg] = (
                image_np[:, :, c][seg] * 0.45 + color[c] * 0.55
            )
    Image.fromarray(overlay_all.astype(np.uint8)).save(
        rdir / "02_all_masks_overlay.png"
    )

    # ── 03 planar masks (green) ───────────────────────────────────────────
    overlay_planar = image_np.copy().astype(np.float64)
    planar_count = 0
    for (eff_mask, ratio, status, _ra) in evaluated_records:
        if status == 'planar':
            planar_count += 1
            for c, val in enumerate((50, 220, 80)):
                overlay_planar[:, :, c][eff_mask] = (
                    image_np[:, :, c][eff_mask] * 0.4 + val * 0.6
                )
    img_planar = overlay_planar.astype(np.uint8)
    cv2.putText(img_planar, f"planar: {planar_count}",
                (6, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_planar, f"planar: {planar_count}",
                (6, 24), font, 0.7, (0, 180, 0), 1, cv2.LINE_AA)
    Image.fromarray(img_planar).save(rdir / "03_planar_overlay.png")

    # ── 04 rejected masks (red) ───────────────────────────────────────────
    overlay_rejected = image_np.copy().astype(np.float64)
    rejected_count = 0
    for (eff_mask, ratio, status, _ra) in evaluated_records:
        if status == 'rejected':
            rejected_count += 1
            for c, val in enumerate((220, 50, 50)):
                overlay_rejected[:, :, c][eff_mask] = (
                    image_np[:, :, c][eff_mask] * 0.4 + val * 0.6
                )
    img_rejected = overlay_rejected.astype(np.uint8)
    cv2.putText(img_rejected, f"rejected: {rejected_count}",
                (6, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_rejected, f"rejected: {rejected_count}",
                (6, 24), font, 0.7, (220, 50, 50), 1, cv2.LINE_AA)
    Image.fromarray(img_rejected).save(rdir / "04_rejected_overlay.png")

    # ── 05 remaining after ────────────────────────────────────────────────
    rem_after_img = (remaining_after.astype(np.uint8) * 255)
    Image.fromarray(rem_after_img, mode='L').save(rdir / "05_remaining_after.png")

    # ── 06 summary grid (5 panels side by side) ───────────────────────────
    panels = [
        np.stack([rem_before_img] * 3, axis=-1),
        overlay_all.astype(np.uint8),
        img_planar,
        img_rejected,
        np.stack([rem_after_img] * 3, axis=-1),
    ]
    panel_labels = [
        "01 remaining_before",
        "02 all_masks",
        "03 planar",
        "04 rejected",
        "05 remaining_after",
    ]
    target_h = min(h, 400)
    scale = target_h / h
    target_w = int(w * scale)
    resized_panels = [
        cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for p in panels
    ]
    for panel, label in zip(resized_panels, panel_labels):
        cv2.putText(panel, label, (4, 18), font, 0.42, (255, 255, 80), 1, cv2.LINE_AA)
    grid = np.concatenate(resized_panels, axis=1)
    Image.fromarray(grid).save(rdir / "06_summary_grid.png")

    # ── 07 本轮确认平面纯可视化 ────────────────────────────────────────
    round_planar_vis = np.zeros((h, w, 3), dtype=np.uint8)
    round_planar_masks = [
        eff_mask for (eff_mask, _ratio, status, _ra) in evaluated_records
        if status == 'planar'
    ]
    for i, mask in enumerate(round_planar_masks):
        color = PALETTE[i % len(PALETTE)]
        for c in range(3):
            round_planar_vis[:, :, c][mask] = color[c]
    Image.fromarray(round_planar_vis).save(rdir / f"07_{round_name}_planar.png")
    Image.fromarray(round_planar_vis).save(debug_dir / f"{round_name}_planar.png")

    # ── 08 法向聚类区域可视化 ──────────────────────────────────────
    cluster_map_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, cm in enumerate(color_masks_np):
        color = PALETTE[k % len(PALETTE)]
        for c in range(3):
            cluster_map_img[:, :, c][cm] = color[c]
    # 混入法向图作为背景纹理以助理解法线方向
    cluster_map_blend = (
        cluster_map_img.astype(np.float64) * 0.65
        + normal_rgb_np.astype(np.float64) * 0.35
    ).clip(0, 255).astype(np.uint8)
    cv2.putText(cluster_map_blend, f"clusters: {len(color_masks_np)}",
                (6, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(cluster_map_blend, f"clusters: {len(color_masks_np)}",
                (6, 24), font, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
    Image.fromarray(cluster_map_blend).save(rdir / "08_cluster_map.png")
    root_cluster_map = debug_dir / "cluster_map.png"
    if not root_cluster_map.exists():
        Image.fromarray(cluster_map_blend).save(root_cluster_map)
    for stale_name in (
        "semantic_cluster_map.png",
        "instance_cluster_map.png",
        "part_cluster_map.png",
    ):
        stale_path = debug_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # ── 09 normal edge 可视化 ─────────────────────────────────────────
    normal_edge_img = (normal_edge_np.astype(np.uint8) * 255)
    Image.fromarray(normal_edge_img, mode='L').save(rdir / "09_normal_edge.png")
    root_normal_edge = debug_dir / "normal_edge.png"
    if not root_normal_edge.exists():
        Image.fromarray(normal_edge_img, mode='L').save(root_normal_edge)
    for stale_name in (
        "semantic_normal_edge.png",
        "instance_normal_edge.png",
        "part_normal_edge.png",
    ):
        stale_path = debug_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    # ── per-mask evaluated images ──────────────────────────────────────────
    for j, (eff_mask, ratio, status, raw_areas) in enumerate(evaluated_records):
        canvas = image_np.copy().astype(np.float64)
        if status == 'planar':
            color_rgb = (50, 220, 80)
        elif status == 'rejected':
            color_rgb = (220, 50, 50)
        else:
            color_rgb = (180, 180, 180)

        for c, val in enumerate(color_rgb):
            canvas[:, :, c][eff_mask] = canvas[:, :, c][eff_mask] * 0.4 + val * 0.6

        canvas_u8 = canvas.astype(np.uint8)

        # 绘制轮廓
        contours, _ = cv2.findContours(
            eff_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas_u8, contours, -1, (255, 255, 0), 1)

        label_text = f"{status.upper()}  ratio={ratio:.3f}  n_clusters={len(raw_areas)}"
        cv2.putText(canvas_u8, label_text, (6, 24), font, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas_u8, label_text, (6, 24), font, 0.65,
                    (0, 0, 0), 1, cv2.LINE_AA)

        fname = f"mask_{j:03d}_ratio{ratio:.3f}_{status.upper()}.png"
        Image.fromarray(canvas_u8).save(masks_dir / fname)

        # 在聚类图上叠加当前 mask，展示每个聚类的重叠像素数
        cluster_canvas = cluster_map_blend.copy().astype(np.float64)
        for c in range(3):
            cluster_canvas[:, :, c][eff_mask] = (
                cluster_canvas[:, :, c][eff_mask] * 0.35 + 255 * 0.65
            )
        cluster_u8 = cluster_canvas.astype(np.uint8)
        cv2.drawContours(cluster_u8, contours, -1, (255, 255, 0), 2)
        # 打印每个聚类的重叠像素数（降序）
        for ki, area_val in enumerate(raw_areas[:8]):
            cv2.putText(cluster_u8, f"c{ki}: {area_val}px",
                        (6, 24 + ki * 22), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(cluster_u8, f"c{ki}: {area_val}px",
                        (6, 24 + ki * 22), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        # 标注 ratio 和判定结果
        verdict = f"ratio={ratio:.3f} -> {status.upper()}"
        cv2.putText(cluster_u8, verdict,
                    (6, 24 + min(len(raw_areas), 8) * 22 + 10), font, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(cluster_u8, verdict,
                    (6, 24 + min(len(raw_areas), 8) * 22 + 10), font, 0.6,
                    color_rgb, 1, cv2.LINE_AA)
        cluster_fname = fname.replace('.png', '_on_clusters.png')
        Image.fromarray(cluster_u8).save(masks_dir / cluster_fname)

    log(f"  [DEBUG] 已保存 round debug → {rdir.relative_to(debug_dir.parent)}", quiet=quiet)
    log(
        f"          all_masks={len(all_masks)}, planar={planar_count}, "
        f"rejected={rejected_count}, "
        f"skipped={sum(1 for r in evaluated_records if r[2] == 'skipped')}",
        quiet=quiet,
    )
    log(f"          累计确认平面: {len(cumulative_planes)}", quiet=quiet)


# ========================= 迭代检测主逻辑 =========================

def detect_planes_iterative(
    model,
    image: Image.Image,
    normal_map: np.ndarray,
    area_threshold: float = AREA_THRESHOLD,
    cluster: int = CLUSTER,
    min_pixels: int = MIN_PLANE_PIXELS,
    text_size: int = 640,
    debug_dir: Path = None,
    quiet: bool = False,
) -> list:
    """
    三轮逐级递进平面检测。

    每轮：
        1. 使用当前粒度生成 masks
        2. 仅保留与 remaining 无重叠（或极少重叠）的 mask
        3. 预计算全图法向聚类
        4. 对每个 mask 先按主导法向簇切成 1~2 个子块
        5. 对子块做法向平坦性筛选，保留更像平面的部分
        6. 确认的平面子块从 remaining 中剔除

    Args:
        area_threshold: 主导法向簇的分裂阈値（默认 0.7）
        cluster:        法向聚类中心数（默认 3）
        debug_dir:      若不为 None，每轮将中间结果保存到该目录
    """
    orig_w, orig_h = image.size
    h, w = orig_h, orig_w
    image_np = np.asarray(image)

    remaining = np.ones((h, w), dtype=bool)
    confirmed_planes = []
    total_pixels = h * w
    min_plane_area = max(min_pixels, int(np.ceil(total_pixels * MIN_PLANE_AREA_RATIO)))
    normal_tensor = torch.from_numpy(normal_map).cuda()
    normal_tensor = normal_tensor / torch.linalg.norm(
        normal_tensor, dim=-1, keepdim=True
    ).clamp(min=1e-8)
    normal_rgb = (normal_tensor + 1.0) * 127.5
    color_masks = SplitPic(
        normal_rgb,
        cluster,
        merge_clusters=True,
    )
    color_masks_np = [m.cpu().numpy().astype(bool) for m in color_masks]
    normal_rgb_np = normal_rgb.cpu().numpy().clip(0, 255).astype(np.uint8)
    normal_edge = compute_normal_edge_map(normal_map)
    log(f"  normal_edge 像素数: {int(normal_edge.sum())}", quiet=quiet)

    for round_idx, round_cfg in enumerate(ROUNDS):
        round_name = round_cfg["name"]
        levels = round_cfg["levels"]

        remaining_ratio = remaining.sum() / total_pixels * 100
        log(f"\n{'─' * 70}", quiet=quiet)
        log(f"▶ Round {round_idx + 1}: {round_name}  |  levels={levels}", quiet=quiet)
        log(f"  剩余未处理区域: {remaining_ratio:.1f}%", quiet=quiet)
        log(
            f"  平面最小面积: {min_plane_area}px ({MIN_PLANE_AREA_RATIO * 100:.1f}%)",
            quiet=quiet,
        )

        if remaining.sum() < min_plane_area:
            log(f"  剩余像素不足 {min_plane_area}，跳过", quiet=quiet)
            continue

        remaining_before = remaining.copy()
        round_focus_mask = None if round_idx == 0 else remaining_before

        # 生成掩码
        with torch.no_grad(), torch.cuda.amp.autocast():
            masks, focus_stats = generate_masks_with_levels(
                model,
                image,
                levels,
                text_size,
                focus_mask=round_focus_mask,
            )
        log(f"  生成掩码数: {len(masks)}", quiet=quiet)
        focus_mode = "full-image" if round_focus_mask is None else "remaining-focus"
        focus_extra = ""
        if round_focus_mask is not None:
            focus_extra = f"  |  focus_pixels={focus_stats.get('focus_pixels', 0)}"
            if focus_stats.get("resampled_with_replacement", False):
                focus_extra += "  |  resample=with-replacement"
        log(
            f"  采样点保留: {focus_stats.get('kept_points', 0)} / "
            f"{focus_stats.get('total_points', 0)}  |  {focus_mode}{focus_extra}",
            quiet=quiet,
        )
        if focus_stats.get("fallback_to_full_image", False):
            log(
                "  [WARN] remaining focus 过滤掉全部采样点，已回退到全图采样",
                quiet=quiet,
            )

        masks.sort(key=lambda x: x['area'], reverse=True)

        log(f"  法向聚类数: {len(color_masks)}", quiet=quiet)

        round_count = 0
        evaluated_records = []   # (effective_mask, ratio, status)

        for mask_dict in masks:
            seg = mask_dict['segmentation']

            mask_area = int(seg.sum())
            if mask_area == 0:
                continue

            # 严格去重：只要 mask 有任意像素与已确认平面重叠，整个 mask 丢弃。
            # 不裁剪（不用 seg & remaining），因为裁剪后的子区域不再是完整平面，
            # 会引入假阳性。已确认像素 = ~remaining。
            overlap_pixels = int((seg & ~remaining).sum())
            if overlap_pixels > 50:  # 允许少量重叠（边界像素等），但超过 50 像素就丢弃整个 mask
                continue

            # mask 完全位于未处理区域，直接使用原始 seg
            effective_mask = seg

            if effective_mask.sum() < min_plane_area:
                evaluated_records.append((effective_mask, 0.0, 'skipped', []))
                continue

            split_masks = split_mask_by_normal_edges(
                effective_mask,
                normal_edge,
                min_plane_area,
            )
            if not split_masks:
                split_masks = [effective_mask]

            accepted = 0
            best_ratio = 0.0
            best_raw_areas = []

            for split_mask in split_masks:
                if int(split_mask.sum()) < min_plane_area:
                    continue

                # ===== 从候选子块内部提取更平坦的子平面 =====
                planar_pieces, raw_areas = extract_planar_submasks(
                    split_mask,
                    color_masks,
                    normal_tensor,
                    area_threshold=area_threshold,
                    min_pixels=min_plane_area,
                )

                if raw_areas:
                    split_ratio = raw_areas[0] / float(max(int(split_mask.sum()), 1))
                    if split_ratio > best_ratio:
                        best_ratio = split_ratio
                        best_raw_areas = raw_areas

                if not planar_pieces:
                    continue

                for piece in planar_pieces:
                    piece_mask = np.logical_and(piece['mask'], np.logical_not(normal_edge))
                    piece_mask = largest_connected_component(piece_mask, min_area=min_plane_area)
                    if piece_mask is None:
                        continue

                    piece_mask = fill_mask_holes(piece_mask, area_thresh=FINAL_MASK_HOLE_AREA)
                    piece_mask = largest_connected_component(piece_mask, min_area=min_plane_area)
                    if piece_mask is None or int(piece_mask.sum()) < min_plane_area:
                        continue

                    confirmed_planes.append({
                        'mask': piece_mask.copy(),
                        'round': round_name,
                        'area': int(piece_mask.sum()),
                        'ratio': float(piece['ratio']),
                        'mean_angle': float(piece['mean_angle']),
                        'p95_angle': float(piece['p95_angle']),
                    })
                    remaining[piece_mask] = False
                    round_count += 1
                    accepted += 1
                    evaluated_records.append((
                        piece_mask,
                        piece['ratio'],
                        'planar',
                        raw_areas,
                    ))

            if accepted > 0:
                continue

            evaluated_records.append((effective_mask, best_ratio, 'rejected', best_raw_areas))

        n_rejected = sum(1 for r in evaluated_records if r[2] == 'rejected')
        n_skipped  = sum(1 for r in evaluated_records if r[2] == 'skipped')
        log(
            f"  ✓ 本轮确认平面: {round_count}  |  "
            f"丢弃: {n_rejected}  |  跳过: {n_skipped}",
            quiet=quiet,
        )
        log(f"  ✓ 累计确认平面: {len(confirmed_planes)} 个", quiet=quiet)

        # ── 保存本轮 debug ────────────────────────────────────────────────
        if debug_dir is not None:
            _save_debug_round(
                round_idx, round_name,
                image_np,
                remaining_before,
                remaining.copy(),
                masks,
                evaluated_records,
                confirmed_planes,
                color_masks_np,
                normal_rgb_np,
                normal_edge,
                debug_dir,
                quiet=quiet,
            )

    return confirmed_planes


# ========================= 可视化 =========================

def visualize_planes(
    image_np: np.ndarray,
    confirmed_planes: list,
    mask_output_dir: Path,
    vis_output_dir: Path,
    image_stem: str,
    debug: bool = False,
):
    h, w = image_np.shape[:2]
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    colored = np.zeros((h, w, 3), dtype=np.uint8)
    index_map = np.zeros((h, w), dtype=np.uint16)
    overlay = image_np.copy().astype(np.float64) if debug else None
    individual_dir = None
    if debug:
        individual_dir = vis_output_dir / f"{image_stem}_planes_individual"
        individual_dir.mkdir(exist_ok=True)

    planes_sorted = sorted(confirmed_planes, key=lambda p: p['area'], reverse=True)

    for i, plane in enumerate(planes_sorted):
        mask = plane['mask']
        color = PALETTE[i % len(PALETTE)]
        index_map[mask] = i + 1

        for c in range(3):
            colored[:, :, c][mask] = color[c]

        if debug:
            for c in range(3):
                overlay[:, :, c][mask] = (
                    image_np[:, :, c][mask] * 0.55 + color[c] * 0.45
                )

            single = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(3):
                single[:, :, c][mask] = color[c]
            single_path = individual_dir / f"plane_{i:03d}_{plane['round']}.png"
            Image.fromarray(single).save(single_path)

    colored_path = vis_output_dir / f"{image_stem}_vis.png"
    Image.fromarray(colored).save(colored_path)

    overlay_path = None
    if debug:
        overlay_path = vis_output_dir / f"{image_stem}_overlay.png"
        Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)

    index_path = mask_output_dir / f"{image_stem}.npy"
    np.save(index_path, index_map)

    return colored_path, overlay_path, index_path


# ========================= 主函数 =========================

def load_normal_map(normal_path: Path, quiet: bool = False) -> np.ndarray:
    log(f"\n[INFO] 加载法线图: {normal_path}", quiet=quiet)
    normal_ext = normal_path.suffix.lower()
    if normal_ext == '.npy':
        normal_map = np.load(normal_path).astype(np.float32)
        log(f"       格式: .npy  shape={normal_map.shape}", quiet=quiet)
    else:
        vis_img = np.asarray(Image.open(normal_path).convert('RGB')).astype(np.float32)
        normal_map = vis_img / 255.0 * 2.0 - 1.0
        norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        normal_map = (normal_map / norms).astype(np.float32)
        log(f"       格式: 可视化图像  shape={normal_map.shape}", quiet=quiet)
        log(f"       已从 pixel 反推并重归一化", quiet=quiet)
        sample_norms = np.linalg.norm(normal_map.reshape(-1, 3), axis=1)
        log(
            f"       模长检查: min={sample_norms.min():.4f}, "
            f"max={sample_norms.max():.4f}, mean={sample_norms.mean():.4f}",
            quiet=quiet,
        )
    log(f"       shape={normal_map.shape}, dtype={normal_map.dtype}", quiet=quiet)
    return normal_map


def collect_input_pairs(image_input: Path, normal_input: Path, quiet: bool = False) -> list:
    image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    normal_suffix_priority = ['.npy', '.png', '.jpg', '.jpeg', '.bmp', '.webp']

    if image_input.is_dir() != normal_input.is_dir():
        raise ValueError("--image 和 --normal 必须同时是文件，或同时是文件夹。")

    if image_input.is_file():
        return [(image_input, normal_input)]

    image_files = {
        p.stem: p
        for p in sorted(image_input.iterdir())
        if p.is_file() and p.suffix.lower() in image_suffixes
    }

    normal_files = {}
    for p in sorted(normal_input.iterdir()):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in normal_suffix_priority:
            continue
        stem = p.stem
        if stem not in normal_files:
            normal_files[stem] = p
            continue
        old_idx = normal_suffix_priority.index(normal_files[stem].suffix.lower())
        new_idx = normal_suffix_priority.index(suffix)
        if new_idx < old_idx:
            normal_files[stem] = p

    common_stems = sorted(set(image_files) & set(normal_files))
    missing_normals = sorted(set(image_files) - set(normal_files))
    missing_images = sorted(set(normal_files) - set(image_files))

    if missing_normals:
        log(
            f"[WARN] 以下 image 没有匹配到 normal，已跳过: {missing_normals[:10]}",
            quiet=quiet,
        )
    if missing_images:
        log(
            f"[WARN] 以下 normal 没有匹配到 image，已跳过: {missing_images[:10]}",
            quiet=quiet,
        )
    if not common_stems:
        raise ValueError("输入文件夹中没有找到同 stem 的 image/normal 配对。")

    return [(image_files[stem], normal_files[stem]) for stem in common_stems]


def process_single_image(
    model,
    image_path: Path,
    normal_path: Path,
    mask_output_dir: Path,
    vis_output_dir: Path,
    area_threshold: float,
    cluster: int,
    min_pixels: int,
    text_size: int,
    debug_enabled: bool,
    quiet: bool,
):
    normal_map = load_normal_map(normal_path, quiet=quiet)

    log(f"[INFO] 加载图像:   {image_path}", quiet=quiet)
    image = Image.open(image_path).convert('RGB')
    image_np = np.asarray(image)
    h, w = image_np.shape[:2]
    log(f"       尺寸: {w} x {h}", quiet=quiet)

    assert normal_map.shape[0] == h and normal_map.shape[1] == w, (
        f"法线图尺寸 {normal_map.shape[:2]} 与图像尺寸 ({h}, {w}) 不匹配！"
    )

    debug_dir = vis_output_dir / f"{image_path.stem}_debug" if debug_enabled else None

    log(f"\n[INFO] 开始多粒度平面检测", quiet=quiet)
    log(
        f"       area_threshold={area_threshold}, cluster={cluster}, min_pixels={min_pixels}",
        quiet=quiet,
    )
    if debug_enabled:
        log(f"       debug 输出: {debug_dir}", quiet=quiet)

    confirmed_planes = detect_planes_iterative(
        model,
        image,
        normal_map,
        area_threshold=area_threshold,
        cluster=cluster,
        min_pixels=min_pixels,
        text_size=text_size,
        debug_dir=debug_dir,
        quiet=quiet,
    )

    log(f"\n{'─' * 70}", quiet=quiet)
    log(f"[INFO] 生成最终可视化...", quiet=quiet)
    colored_path, overlay_path, index_path = visualize_planes(
        image_np,
        confirmed_planes,
        mask_output_dir,
        vis_output_dir,
        image_stem=image_path.stem,
        debug=debug_enabled,
    )

    total_pixels = h * w
    total_plane_pixels = sum(p['area'] for p in confirmed_planes)

    print(f"\n{'=' * 70}")
    print(f"  📊 检测结果统计")
    print(f"{'=' * 70}\n")
    print(f"  检测到平面数: {len(confirmed_planes)}")
    print(f"  平面总覆盖率: {total_plane_pixels / total_pixels * 100:.1f}%\n")

    round_stats = {}
    for p in confirmed_planes:
        r = p['round']
        round_stats.setdefault(r, {'count': 0, 'area': 0})
        round_stats[r]['count'] += 1
        round_stats[r]['area'] += p['area']

    print(f"  {'轮次':<12s} {'平面数':>6s} {'覆盖率':>8s}")
    print(f"  {'─' * 30}")
    for r_name in ['semantic', 'instance', 'part']:
        if r_name in round_stats:
            s = round_stats[r_name]
            print(f"  {r_name:<12s} {s['count']:>6d} {s['area']/total_pixels*100:>7.1f}%")

    print(f"\n  各平面详情 (按面积排序):")
    planes_sorted = sorted(confirmed_planes, key=lambda p: p['area'], reverse=True)
    for i, p in enumerate(planes_sorted):
        print(
            f"    #{i:03d}  {p['round']:<10s}  "
            f"area={p['area']:>8d}px ({p['area']/total_pixels*100:>5.1f}%)  "
            f"ratio={p['ratio']:.3f}  "
            f"mean={p.get('mean_angle', 0.0):>5.1f}deg  "
            f"p95={p.get('p95_angle', 0.0):>5.1f}deg"
        )

    print(f"\n  输出文件:")
    print(f"    mask:    {index_path}")
    print(f"    vis:     {colored_path}")
    if debug_enabled and overlay_path is not None:
        print(f"    overlay: {overlay_path}")
        print(f"    indiv:   {vis_output_dir / f'{image_path.stem}_planes_individual'}/")
        print(f"    debug:   {debug_dir}/")
    print(f"\n{'=' * 70}\n")

    return {
        "image": image_path,
        "normal": normal_path,
        "mask_path": index_path,
        "vis_path": colored_path,
        "overlay_path": overlay_path,
        "num_planes": len(confirmed_planes),
    }

def main():
    seed_everything(RANDOM_SEED)

    parser = argparse.ArgumentParser(
        description="Multi-Granularity Plane Detection via Dual-Center Cosine Distance"
    )
    parser.add_argument(
        "--image", type=str,
        default=str(PROJECT_ROOT / "examples" / "00000.jpg"),
        help="输入图像路径",
    )
    parser.add_argument(
        "--normal", type=str, required=True,
        help="法线图路径，支持两种格式："
             "(1) .npy 文件 (H,W,3) 已归一化单位法线；"
             "(2) 可视化图像 (PNG/JPG)，由 pixel=(normal+1)/2*255 编码",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "outputs_plane_detection"),
        help="输出目录",
    )
    parser.add_argument(
        "--area-threshold", type=float, default=AREA_THRESHOLD,
        dest="area_threshold",
        help=f"法向主导性阈値（0~1），默认 {AREA_THRESHOLD}",
    )
    parser.add_argument(
        "--cluster", type=int, default=CLUSTER,
        help=f"法向聚类中心数，默认 {CLUSTER}",
    )
    parser.add_argument(
        "--min-pixels", type=int, default=MIN_PLANE_PIXELS,
        help=f"mask 最小像素数阈值，默认 {MIN_PLANE_PIXELS}",
    )
    parser.add_argument(
        "--text-size", type=int, default=640,
        help="模型输入缩放尺寸，默认 640",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="保存全部中间结果和调试可视化；默认只保存最终 npy 和 colored",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式：屏蔽全部 print 输出，仅显示 tqdm 进度条",
    )
    args = parser.parse_args()
    apply_quiet_mode(args.quiet)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir = output_dir / "planarmask"
    vis_output_dir = output_dir / "planar_vis"
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    debug_enabled = args.debug and not args.no_debug
    image_input = Path(args.image)
    normal_input = Path(args.normal)

    log("=" * 70, quiet=args.quiet)
    log("  Plane Detection: Normal-Cluster Dominance Test", quiet=args.quiet)
    log("=" * 70, quiet=args.quiet)

    # ---- 加载模型 ----
    log(f"\n[INFO] 加载 Semantic-SAM 模型...", quiet=args.quiet)
    opt = load_opt_from_config_file(CFG_PATH)
    model = (
        BaseModel(opt, build_model(opt))
        .from_pretrained(str(CKPT_PATH))
        .eval()
        .cuda()
    )
    log(f"       ✓ 模型就绪", quiet=args.quiet)

    input_pairs = collect_input_pairs(image_input, normal_input, quiet=args.quiet)
    if image_input.is_dir():
        log(f"\n[INFO] 批量处理模式", quiet=args.quiet)
        log(f"       image dir:  {image_input}", quiet=args.quiet)
        log(f"       normal dir: {normal_input}", quiet=args.quiet)
        log(f"       配对数量:    {len(input_pairs)}", quiet=args.quiet)

    results = []
    input_iter = progress(
        enumerate(input_pairs, start=1),
        total=len(input_pairs),
        quiet=args.quiet,
        desc="Images",
    )
    for idx, (image_path, normal_path) in input_iter:
        input_iter.set_postfix(image=image_path.stem)
        log(f"\n{'#' * 70}", quiet=args.quiet)
        log(f"[INFO] ({idx}/{len(input_pairs)}) 处理 {image_path.stem}", quiet=args.quiet)
        log(f"{'#' * 70}", quiet=args.quiet)
        result = process_single_image(
            model,
            image_path,
            normal_path,
            mask_output_dir,
            vis_output_dir,
            area_threshold=args.area_threshold,
            cluster=args.cluster,
            min_pixels=args.min_pixels,
            text_size=args.text_size,
            debug_enabled=debug_enabled,
            quiet=args.quiet,
        )
        results.append(result)

    print(f"\n{'=' * 70}")
    print("  全部处理完成")
    print(f"{'=' * 70}")
    print(f"  npy 目录: {mask_output_dir}")
    print(f"  vis 目录: {vis_output_dir}")
    print(f"  处理数量: {len(results)}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()

# 单张图像
# python plane_detection.py \
#     --image /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/images/frame_00001.jpg \
#     --normal /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/stable_normal/normal_vis/frame_00001.png \
#     --output outputs/coffee_room/frame_00001 \
#    --debug

# 批量处理
# python plane_detection.py \
#   --image /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/images \
#   --normal /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/stable_normal/normal_vis \
#   --output outputs/mushroom/coffee_room
