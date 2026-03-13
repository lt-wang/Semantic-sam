#!/usr/bin/env python3
# --------------------------------------------------------
# Edge Aggregation: Multi-granularity Mask Boundary Extraction
# Based on: "Edge Aggregation" concept from Semantic-SAM
# --------------------------------------------------------

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

# 配置
PROJECT_ROOT = Path(__file__).parent
CKPT_PATH = PROJECT_ROOT / "ckpts" / "swinl_only_sam_many2many.pth"
INPUT_IMAGE = PROJECT_ROOT / "examples" / "00000.jpg"
CFG_PATH = "configs/semantic_sam_only_sa-1b_swinL.yaml"

# 粒度级别映射
# 1,2: scene级别    3: instance级别    4,5,6: feature级别
LEVELS = [2, 3, 5]  # 选择：scene (2) + instance (3) + feature (5)
LEVEL_NAMES = {2: "scene", 3: "instance", 5: "feature"}


def laplace_operator(mask: np.ndarray) -> np.ndarray:
    """
    应用拉普拉斯算子提取掩码边界
    τ(M) = {(x, y) | ∆M(x, y) ≠ 0}
    
    Args:
        mask: 二进制掩码 (H, W)，dtype=uint8
    
    Returns:
        边界图 (H, W)，边界处为 255，背景为 0
    """
    mask = mask.astype(np.float32)
    
    # 应用拉普拉斯算子
    laplacian = cv2.Laplacian(mask, cv2.CV_32F)
    
    # 提取边界（拉普拉斯值非零的位置）
    boundary = np.abs(laplacian) > 0.1
    
    return (boundary * 255).astype(np.uint8)


def generate_masks_with_level(model, image: Image.Image, level: int, text_size: int = 640):
    """
    为指定粒度级别生成掩码，并调整到原始图像大小
    
    Args:
        model: Semantic-SAM 模型
        image: 输入图像（PIL Image）
        level: 粒度级别 (1/2/3)
        text_size: 图像缩放尺寸
    
    Returns:
        掩码列表（已调整到原始大小）
    """
    orig_w, orig_h = image.size  # 原始大小
    
    # 转换图像
    t = [transforms.Resize(int(text_size), interpolation=Image.BICUBIC)]
    transform = transforms.Compose(t)
    image_transformed = transform(image)
    image_np = np.asarray(image_transformed)
    image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()
    
    # 生成掩码
    mask_generator = SemanticSamAutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=10,
        level=[level],
    )
    
    outputs = mask_generator.generate(image_tensor)
    
    # 调整掩码到原始图像大小
    for mask_dict in outputs:
        seg = mask_dict['segmentation']
        # 调整掩码尺寸到原始大小
        resized_seg = cv2.resize(
            seg.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        mask_dict['segmentation'] = resized_seg.astype(bool)
    
    return outputs


def main():
    print("=" * 80)
    print("Edge Aggregation: Multi-Granularity Boundary Extraction")
    print("=" * 80)
    
    # 加载模型
    print("\n[INFO] 加载模型...")
    opt = load_opt_from_config_file(CFG_PATH)
    model = (
        BaseModel(opt, build_model(opt))
        .from_pretrained(str(CKPT_PATH))
        .eval()
        .cuda()
    )
    
    # 读取输入图像
    print(f"[INFO] 读取图像: {INPUT_IMAGE}")
    image = Image.open(INPUT_IMAGE).convert('RGB')
    image_np = np.asarray(image)
    h, w = image_np.shape[:2]
    print(f"       尺寸: {image.size}")
    
    # 输出目录
    output_dir = PROJECT_ROOT / "outputs_edge_aggregation"
    output_dir.mkdir(exist_ok=True)
    
    # 为每个粒度级别生成掩码和边界图
    print(f"\n[INFO] 生成多粒度掩码和边界图...")
    print(f"{'─' * 80}")
    
    all_boundaries = {}
    mask_counts = {}
    
    for level in LEVELS:
        level_name = LEVEL_NAMES[level]
        print(f"\n▶ 粒度级别 {level} ({level_name})")
        
        # 生成该粒度的掩码
        masks = generate_masks_with_level(model, image, level, text_size=640)
        mask_counts[level] = len(masks)
        print(f"  生成掩码数: {len(masks)}")
        
        # 合并所有掩码
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for i, mask_dict in enumerate(masks):
            seg = mask_dict['segmentation'].astype(np.uint8)
            merged_mask = np.maximum(merged_mask, seg)
        
        # 提取边界
        print(f"  提取边界...")
        boundary = laplace_operator(merged_mask)
        all_boundaries[level] = boundary
        
        # 保存
        level_dir = output_dir / level_name
        level_dir.mkdir(exist_ok=True)
        
        # 保存合并掩码
        merged_path = level_dir / f"merged_mask_{level_name}.png"
        Image.fromarray(merged_mask * 255, mode='L').save(merged_path)
        
        # 保存边界图
        boundary_path = level_dir / f"boundary_{level_name}.png"
        Image.fromarray(boundary, mode='L').save(boundary_path)
        
        # 保存边界叠加在原图上
        overlay = image_np.copy()
        overlay[boundary > 0] = [0, 255, 0]  # 绿色边界
        overlay_path = level_dir / f"boundary_overlay_{level_name}.png"
        Image.fromarray(overlay).save(overlay_path)
        
        print(f"  ✓ 已保存到: {level_dir}")
    
    # 聚合所有边界
    print(f"\n[INFO] 聚合边界图...")
    print(f"{'─' * 80}")
    
    # 多通道边界聚合 (RGB)：自动为 LEVELS 中的每个级别分配颜色
    aggregated_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    _palette = [
        (255,   0,   0),   # 红
        (  0, 255,   0),   # 绿
        (  0,   0, 255),   # 蓝
        (255, 255,   0),   # 黄
        (255,   0, 255),   # 品红
        (  0, 255, 255),   # 青
    ]
    colors = {level: _palette[i % len(_palette)] for i, level in enumerate(LEVELS)}

    for level, boundary in all_boundaries.items():
        color = colors[level]
        for c in range(3):
            aggregated_rgb[:, :, c] = np.maximum(
                aggregated_rgb[:, :, c],
                (boundary > 0).astype(np.uint8) * color[c]
            )
    
    # 保存聚合边界
    agg_path = output_dir / "boundary_aggregated_rgb.png"
    Image.fromarray(aggregated_rgb).save(agg_path)
    
    # 保存聚合边界叠加在原图上
    aggregated_overlay = (image_np * 0.7 + aggregated_rgb * 0.3).astype(np.uint8)
    overlay_path = output_dir / "boundary_aggregated_overlay.png"
    Image.fromarray(aggregated_overlay).save(overlay_path)
    
    print(f"\n✓ 聚合边界已保存:")
    print(f"  - RGB 边界: {agg_path}")
    print(f"  - 叠加效果: {overlay_path}")
    
    # 统计输出
    print(f"\n{'=' * 80}")
    print("📊 处理统计")
    print(f"{'=' * 80}\n")
    
    for level in LEVELS:
        level_name = LEVEL_NAMES[level]
        count = mask_counts[level]
        print(f"  {level_name:10s} (粒度 {level}): {count:3d} 个掩码")
    
    print(f"\n✅ 所有文件已保存到: {output_dir}\n")


if __name__ == '__main__':
    main()
