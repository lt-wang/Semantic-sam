#!/usr/bin/env python3
# --------------------------------------------------------
# Debug script to inspect mask data format
# --------------------------------------------------------

import torch
import os
from PIL import Image
from pathlib import Path

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from utils.constants import COCO_PANOPTIC_CLASSES
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator

# 配置
PROJECT_ROOT = Path(__file__).parent
CKPT_PATH = PROJECT_ROOT / "ckpts" / "swinl_only_sam_many2many.pth"
INPUT_IMAGE = PROJECT_ROOT / "examples" / "frame_00011.jpg"
CFG_PATH = "configs/semantic_sam_only_sa-1b_swinL.yaml"

# 加载模型
print("[INFO] 加载配置和模型...")
opt = load_opt_from_config_file(CFG_PATH)
model = (
    BaseModel(opt, build_model(opt))
    .from_pretrained(str(CKPT_PATH))
    .eval()
    .cuda()
)

# 读取图片
print(f"[INFO] 读取图片: {INPUT_IMAGE}")
image = Image.open(INPUT_IMAGE).convert('RGB')
print(f"       尺寸: {image.size}")

# 转为张量
from torchvision import transforms
import numpy as np

t = [transforms.Resize(640, interpolation=Image.BICUBIC)]
transform = transforms.Compose(t)
image_transformed = transform(image)
image_np = np.asarray(image_transformed)
image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()

print(f"\n[INFO] 按粒度分别推理...")

# 定义粒度级别
granularities = [
    (1, "semantic_level1"),
    (2, "semantic_level2"),
    (3, "instance_level1"),
    (4, "instance_level2"),
    (5, "part_level1"),
    (6, "part_level2"),
]

save_dir = PROJECT_ROOT / "outputs_debug_masks"
save_dir.mkdir(exist_ok=True)

h, w = image_np.shape[:2]
all_results = {}  # 存储各粒度的结果

print(f"\n{'=' * 80}")
print(f"按粒度推理掩码")
print(f"{'=' * 80}\n")

for level, name in granularities:
    if isinstance(level, int):
        level = [level]
    
    print(f"▶ 推理粒度 {name} (level={level})...")
    
    # 生成掩码
    mask_generator = SemanticSamAutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=10,
        level=level,
    )
    
    outputs = mask_generator.generate(image_tensor)
    all_results[name] = {
        'level': level,
        'masks': outputs,
        'count': len(outputs),
    }
    
    print(f"  ✓ 生成掩码数: {len(outputs)}")

# 打印掩码结构（仅第一个粒度）
first_key = list(all_results.keys())[0]
outputs = all_results[first_key]['masks']

print(f"\n{'=' * 80}")
print(f"掩码数据结构分析 (基于 {first_key})")
print(f"{'=' * 80}")
print(f"\n前 5 个掩码的信息:\n")

if len(outputs) > 0:
    for i, mask_dict in enumerate(outputs[:5]):
        print(f"[掩码 {i}]")
        print(f"  类型: {type(mask_dict)}")
        print(f"  键: {list(mask_dict.keys())}")
        
        for key, value in mask_dict.items():
            if key == 'segmentation':
                print(f"  '{key}':")
                print(f"    - 类型: {type(value)}")
                print(f"    - shape: {value.shape}")
                print(f"    - dtype: {value.dtype}")
                print(f"    - 取值范围: [{value.min()}, {value.max()}]")
            elif key == 'area':
                print(f"  '{key}': {value}")
            elif key == 'predicted_iou':
                print(f"  '{key}': {value:.4f}")
            elif key == 'stability_score':
                print(f"  '{key}': {value:.4f}")
            else:
                val_str = str(value)[:60]
                print(f"  '{key}': {val_str}")
        print()

# 对每个粒度保存结果
print(f"\n{'=' * 80}")
print(f"保存各粒度的掩码结果")
print(f"{'=' * 80}\n")

import matplotlib.pyplot as plt
from matplotlib import cm

# 创建对比可视化
granularity_names = list(all_results.keys())
n_granularities = len(granularity_names)
fig, axes = plt.subplots(2, n_granularities, figsize=(4*n_granularities, 8))

if n_granularities == 1:
    axes = axes.reshape(2, 1)

# 第一行：原图（左）+ 掩码统计信息（右）
axes[0, 0].imshow(image_np)
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# 打印统计信息
stat_text = "粒度统计：\n" + "─" * 30 + "\n"
for i, name in enumerate(granularity_names):
    count = all_results[name]['count']
    stat_text += f"{name}: {count} masks\n"

axes[0, 1].text(0.1, 0.5, stat_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0, 1].axis('off')

# 第二行：各粒度的掩码可视化
for idx, name in enumerate(granularity_names):
    result = all_results[name]
    outputs = result['masks']
    n_masks = len(outputs)
    
    # 创建索引掩码
    index_mask = np.zeros((h, w), dtype=np.uint32)
    for i, mask_dict in enumerate(outputs):
        seg = mask_dict['segmentation'].astype(np.uint8)
        index_mask[seg > 0] = i + 1
    
    # 创建彩色掩码
    colormap = cm.get_cmap('hsv')
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for mask_id in range(1, n_masks + 1):
        color_idx = (mask_id - 1) / max(n_masks - 1, 1)
        color = colormap(color_idx)[:3]
        color_uint8 = (np.array(color) * 255).astype(np.uint8)
        mask_region = index_mask == mask_id
        colored_mask[mask_region] = color_uint8
    
    # 显示
    axes[1, idx].imshow(colored_mask)
    axes[1, idx].set_title(f'{name}\n({n_masks} masks)', fontsize=11, fontweight='bold')
    axes[1, idx].axis('off')
    
    # 保存该粒度的结果
    granular_dir = save_dir / name
    granular_dir.mkdir(exist_ok=True)
    
    # 保存单个掩码
    masks_dir = granular_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    for i, mask_dict in enumerate(outputs):
        seg = mask_dict['segmentation'].astype(np.uint8)
        mask_img = Image.fromarray((seg * 255).astype(np.uint8), mode='L')
        mask_img.save(masks_dir / f"mask_{i+1:03d}.png")
    
    # 保存索引掩码
    np.save(granular_dir / "index_mask.npy", index_mask)
    Image.fromarray(colored_mask, mode='RGB').save(granular_dir / "colored_masks.png")
    
    print(f"  ✓ {name}: {n_masks} masks → {granular_dir}")

plt.tight_layout()
comparison_path = save_dir / "granularities_comparison.png"
fig.savefig(comparison_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"\n✓ 对比可视化已保存: {comparison_path}")

# 生成统计报告
print(f"\n{'=' * 80}")
print(f"✅ 所有结果已保存到: {save_dir}")
print(f"{'=' * 80}\n")

for name in granularity_names:
    granular_dir = save_dir / name
    count = all_results[name]['count']
    print(f"  📁 {name}/")
    print(f"     ├─ masks/ ({count} 个 mask PNG 文件)")
    print(f"     ├─ index_mask.npy (掩码索引图)")
    print(f"     └─ colored_masks.png (彩色可视化)")

print(f"\n  📊 granularities_comparison.png (对比总览)\n")
