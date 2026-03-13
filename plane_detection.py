#!/usr/bin/env python3
# --------------------------------------------------------
# Plane Detection: Multi-Granularity Planar Region Detection
# via Normal-Cluster Dominance Test
# --------------------------------------------------------

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from torchvision import transforms

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
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
    {"name": "part",     "levels": [5]},
]
AREA_THRESHOLD = 0.7
CLUSTER = 5
MIN_PLANE_PIXELS = 200
MIN_PLANE_AREA_RATIO = 0.005
NORMAL_MERGE_COLOR_THRESH = 50.0
MAX_PLANE_MEAN_ANGLE = 13.0
MAX_PLANE_P95_ANGLE = 30.0

PALETTE = [
    (230,  25,  75), (60,  180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


# ========================= 法向聚类工具函数 =========================

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
    indices = torch.randperm(N, device=device)[:num_clusters]
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

def generate_masks_with_levels(
    model, image: Image.Image, levels: list, text_size: int = 640
) -> list:
    orig_w, orig_h = image.size
    t = [transforms.Resize(int(text_size), interpolation=Image.BICUBIC)]
    transform = transforms.Compose(t)
    image_transformed = transform(image)
    image_np = np.asarray(image_transformed)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

    mask_generator = SemanticSamAutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=10,
        level=levels,
    )
    outputs = mask_generator.generate(image_tensor)

    for mask_dict in outputs:
        seg = mask_dict['segmentation']
        resized = cv2.resize(
            seg.astype(np.uint8), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_dict['segmentation'] = resized.astype(bool)
    return outputs


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
    debug_dir: Path,
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
        07_{round_name}_planar.png — 累计已确认平面（含之前所有轮次）
        08_cluster_map.png         — 本轮法向聚类区域可视化
        all_masks/                 — 本轮生成器输出的所有原始 mask
            mask_{j:03d}.png
        evaluated/
            mask_{j:03d}_ratio{:.3f}_{PLANAR|REJECTED|SKIPPED}.png
    """
    h, w = image_np.shape[:2]
    rdir = debug_dir / f"round_{round_idx + 1:02d}_{round_name}"
    rdir.mkdir(parents=True, exist_ok=True)
    masks_dir = rdir / "evaluated"
    masks_dir.mkdir(exist_ok=True)

    # ── 保存所有原始 mask ──────────────────────────────────────────────
    all_masks_dir = rdir / "all_masks"
    all_masks_dir.mkdir(exist_ok=True)
    for j, md in enumerate(all_masks):
        seg = md['segmentation']
        mask_img = (seg.astype(np.uint8) * 255)
        Image.fromarray(mask_img, mode='L').save(all_masks_dir / f"mask_{j:03d}.png")

    font = cv2.FONT_HERSHEY_SIMPLEX

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

    # ── 07 累计已确认平面（含之前所有轮次） ─────────────────────────────
    overlay_cumulative = image_np.copy().astype(np.float64)
    for i, plane in enumerate(cumulative_planes):
        mask = plane['mask']
        color = PALETTE[i % len(PALETTE)]
        for c in range(3):
            overlay_cumulative[:, :, c][mask] = (
                image_np[:, :, c][mask] * 0.45 + color[c] * 0.55
            )
    img_cumulative = overlay_cumulative.astype(np.uint8)
    total_cumulative = len(cumulative_planes)
    cv2.putText(img_cumulative, f"{round_name}_planar: {total_cumulative} planes (cumulative)",
                (6, 24), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_cumulative, f"{round_name}_planar: {total_cumulative} planes (cumulative)",
                (6, 24), font, 0.7, (0, 200, 255), 1, cv2.LINE_AA)
    Image.fromarray(img_cumulative).save(rdir / f"07_{round_name}_planar.png")
    # 同时保存一份到 debug 根目录，方便查看
    Image.fromarray(img_cumulative).save(debug_dir / f"{round_name}_planar.png")

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
    Image.fromarray(cluster_map_blend).save(debug_dir / f"{round_name}_cluster_map.png")

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

    print(f"  [DEBUG] 已保存 round debug → {rdir.relative_to(debug_dir.parent)}")
    print(f"          all_masks={len(all_masks)}, planar={planar_count}, "
          f"rejected={rejected_count}, "
          f"skipped={sum(1 for r in evaluated_records if r[2] == 'skipped')}")
    print(f"          累计确认平面: {total_cumulative}")


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
        merge_clusters=False,
    )
    color_masks_np = [m.cpu().numpy().astype(bool) for m in color_masks]
    normal_rgb_np = normal_rgb.cpu().numpy().clip(0, 255).astype(np.uint8)

    for round_idx, round_cfg in enumerate(ROUNDS):
        round_name = round_cfg["name"]
        levels = round_cfg["levels"]

        remaining_ratio = remaining.sum() / total_pixels * 100
        print(f"\n{'─' * 70}")
        print(f"▶ Round {round_idx + 1}: {round_name}  |  levels={levels}")
        print(f"  剩余未处理区域: {remaining_ratio:.1f}%")
        print(f"  平面最小面积: {min_plane_area}px ({MIN_PLANE_AREA_RATIO * 100:.1f}%)")

        if remaining.sum() < min_plane_area:
            print(f"  剩余像素不足 {min_plane_area}，跳过")
            continue

        remaining_before = remaining.copy()

        # 生成掩码
        with torch.no_grad(), torch.cuda.amp.autocast():
            masks = generate_masks_with_levels(model, image, levels, text_size)
        print(f"  生成掩码数: {len(masks)}")

        masks.sort(key=lambda x: x['area'], reverse=True)

        print(f"  法向聚类数: {len(color_masks)}")

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

            # ===== 从候选 mask 内部提取更平坦的子平面 =====
            planar_pieces, raw_areas = extract_planar_submasks(
                effective_mask,
                color_masks,
                normal_tensor,
                area_threshold=area_threshold,
                min_pixels=min_plane_area,
            )

            if planar_pieces:
                accepted = 0
                for piece in planar_pieces:
                    piece_mask = piece['mask']
                    if int(piece_mask.sum()) < min_plane_area:
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

            best_ratio = (raw_areas[0] / float(mask_area)) if raw_areas and mask_area > 0 else 0.0
            evaluated_records.append((effective_mask, best_ratio, 'rejected', raw_areas))

        n_rejected = sum(1 for r in evaluated_records if r[2] == 'rejected')
        n_skipped  = sum(1 for r in evaluated_records if r[2] == 'skipped')
        print(f"  ✓ 本轮确认平面: {round_count}  |  "
              f"丢弃: {n_rejected}  |  跳过: {n_skipped}")
        print(f"  ✓ 累计确认平面: {len(confirmed_planes)} 个")

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
                debug_dir,
            )

    return confirmed_planes


# ========================= 可视化 =========================

def visualize_planes(
    image_np: np.ndarray,
    confirmed_planes: list,
    output_dir: Path,
):
    h, w = image_np.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    colored = np.zeros((h, w, 3), dtype=np.uint8)
    overlay = image_np.copy().astype(np.float64)

    individual_dir = output_dir / "planes_individual"
    individual_dir.mkdir(exist_ok=True)

    planes_sorted = sorted(confirmed_planes, key=lambda p: p['area'], reverse=True)

    for i, plane in enumerate(planes_sorted):
        mask = plane['mask']
        color = PALETTE[i % len(PALETTE)]

        for c in range(3):
            colored[:, :, c][mask] = color[c]

        for c in range(3):
            overlay[:, :, c][mask] = (
                image_np[:, :, c][mask] * 0.55 + color[c] * 0.45
            )

        single = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            single[:, :, c][mask] = color[c]
        single_path = individual_dir / f"plane_{i:03d}_{plane['round']}.png"
        Image.fromarray(single).save(single_path)

    colored_path = output_dir / "planes_colored.png"
    Image.fromarray(colored).save(colored_path)

    overlay_path = output_dir / "planes_overlay.png"
    Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)

    return colored_path, overlay_path


# ========================= 主函数 =========================

def main():
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
        "--no-debug", action="store_true",
        help="禁用每轮中间结果的 debug 可视化输出",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None if args.no_debug else (output_dir / "debug")

    print("=" * 70)
    print("  Plane Detection: Normal-Cluster Dominance Test")
    print("=" * 70)

    # ---- 加载法线图 ----
    print(f"\n[INFO] 加载法线图: {args.normal}")
    normal_ext = Path(args.normal).suffix.lower()
    if normal_ext == '.npy':
        # 格式 1：直接加载已归一化法线
        normal_map = np.load(args.normal).astype(np.float32)  # (H, W, 3)
        print(f"       格式: .npy  shape={normal_map.shape}")
    else:
        # 格式 2：从可视化图像反推法线
        # 可视化公式： pixel = (normal + 1.0) * 0.5 * 255
        # 反推公式： normal = pixel / 255.0 * 2.0 - 1.0
        vis_img = np.asarray(Image.open(args.normal).convert('RGB')).astype(np.float32)
        normal_map = vis_img / 255.0 * 2.0 - 1.0  # 映射回 [-1, 1]
        # JPEG 有损压缩会微小改变数候，重归一化保证是单位向量
        norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)  # 防止除零
        normal_map = (normal_map / norms).astype(np.float32)
        print(f"       格式: 可视化图像  shape={normal_map.shape}")
        print(f"       已从 pixel 反推并重归一化")
        # 输出简单检查
        sample_norms = np.linalg.norm(normal_map.reshape(-1,3), axis=1)
        print(f"       模长检查: min={sample_norms.min():.4f}, "
              f"max={sample_norms.max():.4f}, mean={sample_norms.mean():.4f}")
    print(f"       shape={normal_map.shape}, dtype={normal_map.dtype}")

    # ---- 加载图像 ----
    print(f"[INFO] 加载图像:   {args.image}")
    image = Image.open(args.image).convert('RGB')
    image_np = np.asarray(image)
    h, w = image_np.shape[:2]
    print(f"       尺寸: {w} x {h}")

    assert normal_map.shape[0] == h and normal_map.shape[1] == w, (
        f"法线图尺寸 {normal_map.shape[:2]} 与图像尺寸 ({h}, {w}) 不匹配！"
    )

    # ---- 加载模型 ----
    print(f"\n[INFO] 加载 Semantic-SAM 模型...")
    opt = load_opt_from_config_file(CFG_PATH)
    model = (
        BaseModel(opt, build_model(opt))
        .from_pretrained(str(CKPT_PATH))
        .eval()
        .cuda()
    )
    print(f"       ✓ 模型就绪")

    # ---- 迭代检测 ----
    print(f"\n[INFO] 开始多粒度平面检测")
    print(f"       area_threshold={args.area_threshold}, cluster={args.cluster}, min_pixels={args.min_pixels}")
    if debug_dir:
        print(f"       debug 输出: {debug_dir}")

    confirmed_planes = detect_planes_iterative(
        model, image, normal_map,
        area_threshold=args.area_threshold,
        cluster=args.cluster,
        min_pixels=args.min_pixels,
        text_size=args.text_size,
        debug_dir=debug_dir,
    )

    # ---- 可视化 ----
    print(f"\n{'─' * 70}")
    print(f"[INFO] 生成最终可视化...")
    colored_path, overlay_path = visualize_planes(image_np, confirmed_planes, output_dir)

    # ---- 统计报告 ----
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
    print(f"    纯色图:  {colored_path}")
    print(f"    叠加图:  {overlay_path}")
    print(f"    单独图:  {output_dir / 'planes_individual'}/")
    if debug_dir:
        print(f"    Debug:   {debug_dir}/")
    print(f"\n{'=' * 70}\n")


if __name__ == '__main__':
    main()

# python plane_detection.py \
#     --image /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/images/frame_00011.jpg \
#     --normal /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/stable_normal/normal_vis/frame_00011.png \
#     --output outputs/coffee_room/plane_detection \
#     --area-threshold 0.7 \
#     --cluster 5 \
#     --min-pixels 200
