#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sam_utils.amg import remove_small_regions


def clean_mask(
    mask: np.ndarray,
    area_thresh: int = 500,
    close_kernel: int = 3,
    close_iterations: int = 1,
) -> np.ndarray:
    mask = mask.astype(np.uint8)

    # Close thin cracks before removing regions.
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=close_iterations,
    ).astype(bool)

    mask, _ = remove_small_regions(mask, area_thresh, mode="holes")
    mask, _ = remove_small_regions(mask, area_thresh, mode="islands")
    return mask


def load_npy_mask(npy_path: Path, label: int | None, channel: int | None) -> np.ndarray:
    array = np.load(npy_path, allow_pickle=False)
    array = np.asarray(array)

    if array.ndim == 0:
        raise ValueError(f"Expected an array mask in {npy_path}, got a scalar.")

    array = np.squeeze(array)

    if array.ndim == 3 and channel is not None:
        if channel < 0:
            raise ValueError("--channel must be non-negative.")
        if array.shape[-1] <= 16 and channel < array.shape[-1]:
            array = array[..., channel]
        elif array.shape[0] <= 16 and channel < array.shape[0]:
            array = array[channel]
        else:
            raise ValueError(
                f"Cannot select channel {channel} from array with shape {array.shape}."
            )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask after loading {npy_path}, got shape {array.shape}. "
            "Use --channel for 3D arrays."
        )

    if label is not None:
        return array == label

    if array.dtype == np.bool_:
        return array

    return array > 0


def load_mask(image_path: Path, rgb: tuple[int, int, int] | None, tol: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if image.ndim == 2:
        return image > 127

    if image.shape[2] == 4:
        image = image[:, :, :3]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if rgb is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray > 127

    target = np.array(rgb, dtype=np.int16)
    diff = np.abs(image_rgb.astype(np.int16) - target[None, None, :])
    return np.all(diff <= tol, axis=-1)


def make_synthetic_mask(h: int = 512, w: int = 512) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (40, 240), (470, 500), 1, thickness=-1)

    rng = np.random.default_rng(22)

    for _ in range(120):
        x = int(rng.integers(60, 450))
        y = int(rng.integers(260, 490))
        rx = int(rng.integers(3, 12))
        ry = int(rng.integers(1, 5))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(mask, (x, y), (rx, ry), angle, 0, 360, 0, thickness=-1)

    for _ in range(25):
        x = int(rng.integers(10, 500))
        y = int(rng.integers(10, 220))
        r = int(rng.integers(2, 6))
        cv2.circle(mask, (x, y), r, 1, thickness=-1)

    return mask.astype(bool)


def render_binary(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def render_red_overlay(mask: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    if background is None:
        background = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    canvas = background.copy()
    canvas[mask] = np.array([235, 20, 90], dtype=np.uint8)
    return canvas


def save_outputs(mask_before: np.ndarray, mask_after: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    before_bin = render_binary(mask_before)
    after_bin = render_binary(mask_after)
    before_red = render_red_overlay(mask_before)
    after_red = render_red_overlay(mask_after)

    diff = np.zeros((*mask_before.shape, 3), dtype=np.uint8)
    diff[np.logical_and(mask_before, ~mask_after)] = (255, 140, 0)
    diff[np.logical_and(~mask_before, mask_after)] = (0, 200, 0)
    diff[np.logical_and(mask_before, mask_after)] = (235, 20, 90)

    cv2.imwrite(str(output_dir / "mask_before.png"), before_bin)
    cv2.imwrite(str(output_dir / "mask_after.png"), after_bin)
    cv2.imwrite(str(output_dir / "overlay_before.png"), cv2.cvtColor(before_red, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "overlay_after.png"), cv2.cvtColor(after_red, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "diff.png"), cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test clean_mask on a binary or color mask image.")
    parser.add_argument("--input", type=Path, help="Input image path. Supports image files and .npy masks. If omitted, a synthetic noisy mask is generated.")
    parser.add_argument("--output", type=Path, default=Path("outputs_test/clean_mask"), help="Output directory.")
    parser.add_argument("--rgb", type=int, nargs=3, metavar=("R", "G", "B"), help="Target RGB color for extracting a mask from a color visualization.")
    parser.add_argument("--tol", type=int, default=20, help="Color tolerance used with --rgb.")
    parser.add_argument("--label", type=int, help="Label value to extract from a 2D integer .npy mask. Without this option, non-zero values are treated as foreground.")
    parser.add_argument("--channel", type=int, help="Channel index to use when loading a 3D .npy array.")
    parser.add_argument("--area-thresh", type=int, default=500, help="Small holes / islands area threshold.")
    parser.add_argument("--close-kernel", type=int, default=3, help="Closing kernel size.")
    parser.add_argument("--close-iterations", type=int, default=1, help="Closing iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is None:
        mask_before = make_synthetic_mask()
        source_name = "synthetic"
    else:
        if args.input.suffix.lower() == ".npy":
            mask_before = load_npy_mask(args.input, label=args.label, channel=args.channel)
        else:
            rgb = tuple(args.rgb) if args.rgb is not None else None
            mask_before = load_mask(args.input, rgb=rgb, tol=args.tol)
        source_name = args.input.stem

    mask_after = clean_mask(
        mask_before,
        area_thresh=args.area_thresh,
        close_kernel=args.close_kernel,
        close_iterations=args.close_iterations,
    )

    output_dir = args.output / source_name
    save_outputs(mask_before, mask_after, output_dir)

    before_area = int(mask_before.sum())
    after_area = int(mask_after.sum())
    changed = int(np.logical_xor(mask_before, mask_after).sum())

    print(f"source:  {source_name}")
    print(f"before:  {before_area} px")
    print(f"after:   {after_area} px")
    print(f"changed: {changed} px")
    print(f"saved:   {output_dir}")


if __name__ == "__main__":
    main()
