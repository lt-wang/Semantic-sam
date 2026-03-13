# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import argparse
import os
from PIL import Image

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import interactive_infer_image_idino_m2m_auto


# ── 配置 ────────────────────────────────────────────────────────────────────

CFGS = {
    'T': "configs/semantic_sam_only_sa-1b_swinT.yaml",
    'L': "configs/semantic_sam_only_sa-1b_swinL.yaml",
}

ALL_CLASSES = [
    name.replace('-other', '').replace('-merged', '')
    for name in COCO_PANOPTIC_CLASSES
]

ALL_PARTS = [
    'arm', 'beak', 'body', 'cap', 'door', 'ear', 'eye', 'foot', 'hair',
    'hand', 'handlebar', 'head', 'headlight', 'horn', 'leg', 'license plate',
    'mirror', 'mouth', 'muzzle', 'neck', 'nose', 'paw', 'plant', 'pot',
    'saddle', 'tail', 'torso', 'wheel', 'window', 'wing',
]


# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_option():
    parser = argparse.ArgumentParser(
        description='Semantic-SAM CLI — 自动分割生成',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必填
    parser.add_argument(
        '-i', '--input', required=True,
        metavar='IMAGE',
        help='输入图片路径（支持 PNG / JPG / JPEG 等 PIL 可读格式）',
    )
    parser.add_argument(
        '--ckpt', required=True,
        metavar='FILE',
        help='模型权重文件路径（.pth）',
    )

    # 输出
    parser.add_argument(
        '-o', '--output', default='output.png',
        metavar='FILE',
        help='输出图片保存路径',
    )

    # 模型选择
    parser.add_argument(
        '--model-size', default='L', choices=['T', 'L'],
        metavar='SIZE',
        help='模型规模：T = Swin-Tiny，L = Swin-Large',
    )

    # 粒度 prompt 级别
    parser.add_argument(
        '--level', default='all',
        metavar='LEVEL',
        help=(
            '分割粒度级别。可选值：\n'
            '  all        — 使用全部 6 个粒度 prompt（默认）\n'
            '  1~6        — 单个粒度（1 最粗，6 最细）\n'
            '  1,3,5      — 逗号分隔的多个粒度'
        ),
    )

    # 推理超参
    parser.add_argument(
        '--text-size', type=int, default=640,
        help='图像短边缩放尺寸（像素）',
    )
    parser.add_argument(
        '--hole-scale', type=int, default=100,
        help='掩码空洞过滤阈值',
    )
    parser.add_argument(
        '--island-scale', type=int, default=100,
        help='掩码孤岛过滤阈值',
    )
    parser.add_argument(
        '--thresh', type=str, default='0.0',
        metavar='THRESH',
        help='掩码 IoU 分数过滤阈值（字符串形式，如 "0.5"）',
    )

    args = parser.parse_args()
    return args


# ── level 参数解析 ───────────────────────────────────────────────────────────

def parse_level(level_str: str):
    """
    将 --level 参数字符串转换为模型所需的列表格式。

    'all'   → [1, 2, 3, 4, 5, 6]
    '3'     → ['3']
    '1,3,5' → ['1', '3', '5']
    """
    level_str = level_str.strip().lower()
    if level_str == 'all':
        return [1, 2, 3, 4, 5, 6]
    parts = [p.strip() for p in level_str.split(',')]
    for p in parts:
        if not p.isdigit() or not (1 <= int(p) <= 6):
            raise ValueError(
                f"--level 的值 '{p}' 无效，请使用 'all' 或 1~6 之间的整数（可逗号分隔）。"
            )
    return parts


# ── 推理入口 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def inference(model, image: Image.Image, level, text_size, hole_scale,
              island_scale, thresh):
    """执行自动分割推理，返回可视化结果 PIL Image。"""
    text = ':'.join(ALL_CLASSES)
    text_part = ':'.join(ALL_PARTS)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        result = interactive_infer_image_idino_m2m_auto(
            model, image, level,
            text, text_part, thresh,
            text_size, hole_scale, island_scale,
            semantic=True,
        )
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_option()

    # ── 验证输入文件 ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"找不到输入图片：{args.input}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"找不到模型权重：{args.ckpt}")

    # ── 解析 level ────────────────────────────────────────────────────────────
    level = parse_level(args.level)
    print(f"[INFO] 使用粒度级别: {level}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    cfg_path = CFGS[args.model_size]
    print(f"[INFO] 加载配置: {cfg_path}")
    print(f"[INFO] 加载权重: {args.ckpt}")
    opt = load_opt_from_config_file(cfg_path)
    model = (
        BaseModel(opt, build_model(opt))
        .from_pretrained(args.ckpt)
        .eval()
        .cuda()
    )
    print("[INFO] 模型加载完成。")

    # ── 读取输入图片 ──────────────────────────────────────────────────────────
    image = Image.open(args.input).convert('RGB')
    print(f"[INFO] 输入图片: {args.input}  尺寸: {image.size}")

    # ── 推理 ──────────────────────────────────────────────────────────────────
    print("[INFO] 开始推理...")
    result = inference(
        model, image, level,
        args.text_size, args.hole_scale, args.island_scale, args.thresh,
    )

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    result.save(args.output)
    print(f"[INFO] 结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
