#!/usr/bin/env python3
# --------------------------------------------------------
# Plane Detection: Multi-Granularity Planar Region Detection
# via Dual-Center Cosine Distance Discrimination
# --------------------------------------------------------

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from torchvision import transforms
from sklearn.cluster import KMeans

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator

# ========================= 配置 =========================
PROJECT_ROOT = Path(__file__).parent
CKPT_PATH = PROJECT_ROOT / "ckpts" / "swinl_only_sam_many2many.pth"
CFG_PATH = "configs/semantic_sam_only_sa-1b_swinL.yaml"

ROUNDS = [
    {"name": "semantic", "levels": [1,2]},
    {"name": "instance", "levels": [1, 2, 3]},
    {"name": "part",     "levels": [1, 2, 3, 4, 5, 6]},
]

TAU_PLANAR_DEG = 5.0
MIN_PLANE_PIXELS = 200

PALETTE = [
    (230,  25,  75), (60,  180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


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
    evaluated_records: list,   # list of (effective_mask, theta_deg, status)
                               # status: 'planar' | 'rejected' | 'skipped'
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
        masks/
            mask_{j:03d}_theta{:.1f}_{PLANAR|REJECTED|SKIPPED}.png
    """
    h, w = image_np.shape[:2]
    rdir = debug_dir / f"round_{round_idx + 1:02d}_{round_name}"
    rdir.mkdir(parents=True, exist_ok=True)
    masks_dir = rdir / "masks"
    masks_dir.mkdir(exist_ok=True)

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
    for (eff_mask, theta, status) in evaluated_records:
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
    for (eff_mask, theta, status) in evaluated_records:
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

    # ── per-mask images ────────────────────────────────────────────────────
    for j, (eff_mask, theta, status) in enumerate(evaluated_records):
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

        label_text = f"{status.upper()}  theta={theta:.2f}deg"
        cv2.putText(canvas_u8, label_text, (6, 24), font, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas_u8, label_text, (6, 24), font, 0.65,
                    (0, 0, 0), 1, cv2.LINE_AA)

        fname = f"mask_{j:03d}_theta{theta:.1f}_{status.upper()}.png"
        Image.fromarray(canvas_u8).save(masks_dir / fname)

    print(f"  [DEBUG] 已保存 round debug → {rdir.relative_to(debug_dir.parent)}")
    print(f"          planar={planar_count}, rejected={rejected_count}, "
          f"skipped={sum(1 for _, _, s in evaluated_records if s == 'skipped')}")


# ========================= 迭代检测主逻辑 =========================

def detect_planes_iterative(
    model,
    image: Image.Image,
    normal_map: np.ndarray,
    tau_deg: float = TAU_PLANAR_DEG,
    min_pixels: int = MIN_PLANE_PIXELS,
    text_size: int = 640,
    debug_dir: Path = None,
) -> list:
    """
    三轮逐级递进平面检测。

    每轮：
        1. 使用当前粒度生成 masks
        2. 仅保留与 remaining 重叠率 > 50% 的 mask
        3. 对每个 mask 做平面判别（双中心余弦距离）
        4. 确认的平面从 remaining 中剔除
        5. 非平面丢弃，交由下一轮更细粒度处理

    Args:
        debug_dir: 若不为 None，每轮将中间结果保存到该目录
    """
    orig_w, orig_h = image.size
    h, w = orig_h, orig_w
    image_np = np.asarray(image)

    remaining = np.ones((h, w), dtype=bool)
    confirmed_planes = []
    total_pixels = h * w

    for round_idx, round_cfg in enumerate(ROUNDS):
        round_name = round_cfg["name"]
        levels = round_cfg["levels"]

        remaining_ratio = remaining.sum() / total_pixels * 100
        print(f"\n{'─' * 70}")
        print(f"▶ Round {round_idx + 1}: {round_name}  |  levels={levels}")
        print(f"  剩余未处理区域: {remaining_ratio:.1f}%")

        if remaining.sum() < min_pixels:
            print(f"  剩余像素不足 {min_pixels}，跳过")
            continue

        remaining_before = remaining.copy()

        # 生成掩码
        with torch.no_grad(), torch.cuda.amp.autocast():
            masks = generate_masks_with_levels(model, image, levels, text_size)
        print(f"  生成掩码数: {len(masks)}")

        masks.sort(key=lambda x: x['area'], reverse=True)

        round_count = 0
        evaluated_records = []   # (effective_mask, theta_deg, status)

        for mask_dict in masks:
            seg = mask_dict['segmentation']

            mask_area = int(seg.sum())
            if mask_area == 0:
                continue

            # 严格去重：只要 mask 有任意像素与已确认平面重叠，整个 mask 丢弃。
            # 不裁剪（不用 seg & remaining），因为裁剪后的子区域不再是完整平面，
            # 会引入假阳性。已确认像素 = ~remaining。
            overlap_pixels = int((seg & ~remaining).sum())
            if overlap_pixels > 10:  # 允许少量重叠（边界像素等），但超过 10 像素就丢弃整个 mask
                continue

            # mask 完全位于未处理区域，直接使用原始 seg
            effective_mask = seg

            if effective_mask.sum() < min_pixels:
                evaluated_records.append((effective_mask, 0.0, 'skipped'))
                continue

            # ===== 双中心余弦距离判别 =====
            normals = normal_map[effective_mask]
            # 符号对齐：将所有法线对齐到均値法线所在半球
            # 修复 Dust3r 等重建方法导致的法线符号歧义问题：
            # 同一平面的不同像素法线可能指向 +n 或 -n，
            # K-Means 会把这两组分开，导致 θ 赋180°。
            mean_normal = normals.mean(axis=0)
            mean_normal_norm = np.linalg.norm(mean_normal)
            if mean_normal_norm < 1e-6:
                evaluated_records.append((effective_mask, 0.0, 'skipped'))
                continue
            mean_normal /= mean_normal_norm
            # dot > 0: 同向, dot < 0: 反向→翻转
            dot_signs = np.sign(normals @ mean_normal)  # (N,)
            dot_signs[dot_signs == 0] = 1.0
            normals = normals * dot_signs[:, np.newaxis]  # 对齐到同一半球
            kmeans = KMeans(n_clusters=2, n_init=3, max_iter=50, random_state=42)
            kmeans.fit(normals)

            mu1 = kmeans.cluster_centers_[0]
            mu2 = kmeans.cluster_centers_[1]
            mu1 = mu1 / (np.linalg.norm(mu1) + 1e-8)
            mu2 = mu2 / (np.linalg.norm(mu2) + 1e-8)

            cos_sim = np.clip(np.dot(mu1, mu2), -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_sim))

            if theta_deg < tau_deg:
                confirmed_planes.append({
                    'mask': effective_mask.copy(),
                    'round': round_name,
                    'area': int(effective_mask.sum()),
                    'theta': float(theta_deg),
                })
                remaining[effective_mask] = False
                round_count += 1
                evaluated_records.append((effective_mask, theta_deg, 'planar'))
            else:
                evaluated_records.append((effective_mask, theta_deg, 'rejected'))

        n_rejected = sum(1 for _, _, s in evaluated_records if s == 'rejected')
        n_skipped  = sum(1 for _, _, s in evaluated_records if s == 'skipped')
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
        "--tau", type=float, default=TAU_PLANAR_DEG,
        help=f"平面判定阈值（度），默认 {TAU_PLANAR_DEG}°",
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
    print("  Plane Detection: Dual-Center Cosine Distance Discrimination")
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
    print(f"       τ_planar={args.tau}°, min_pixels={args.min_pixels}")
    if debug_dir:
        print(f"       debug 输出: {debug_dir}")

    confirmed_planes = detect_planes_iterative(
        model, image, normal_map,
        tau_deg=args.tau,
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
            f"θ={p['theta']:.2f}°"
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
#     --tau 5.0 \
#     --min-pixels 200