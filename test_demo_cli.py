#!/usr/bin/env python3
# --------------------------------------------------------
# Semantic-SAM CLI Demo Test
# Test script for demo_cli.py
# --------------------------------------------------------

import subprocess
import os
import sys
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DEMO_CLI = PROJECT_ROOT / "demo_cli.py"
CKPT_PATH = PROJECT_ROOT / "ckpts" / "swinl_only_sam_many2many.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs_test"

# 测试用例：(输入图片, 输出文件名, --level 参数, --model-size 参数)
TEST_CASES = [
    ("examples/dog.jpg", "test_dog_all.png", "all", "L"),
    ("examples/tank.png", "test_tank_1.png", "1", "L"),
    ("examples/castle.png", "test_castle_all.png", "all", "T"),
    ("examples/corgi2.jpg", "test_corgi_1_3_5.png", "1,3,5", "L"),
]


# ── 测试函数 ──────────────────────────────────────────────────────────────────

def run_test(input_image, output_name, level, model_size):
    """运行单个测试用例。"""
    input_path = PROJECT_ROOT / input_image
    output_path = OUTPUT_DIR / output_name

    # 验证输入
    if not input_path.exists():
        print(f"❌ 输入图片不存在: {input_path}")
        return False

    if not CKPT_PATH.exists():
        print(f"❌ 模型权重不存在: {CKPT_PATH}")
        return False

    # 构建命令（需要激活 conda 环境）
    cmd = [
        "bash", "-c",
        (
            f"python {DEMO_CLI} "
            f"--input {input_path} "
            f"--ckpt {CKPT_PATH} "
            f"--output {output_path} "
            f"--level {level} "
            f"--model-size {model_size} "
            f"--text-size 640 "
            f"--hole-scale 100 "
            f"--island-scale 100"
        )
    ]

    print(f"\n{'─' * 80}")
    print(f"🧪 测试: {input_image} → {output_name}")
    print(f"   参数: --level={level} --model-size={model_size}")
    print(f"{'─' * 80}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=300,  # 5 分钟超时
        )

        if result.returncode == 0:
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✅ 成功! 输出文件: {output_path} ({file_size:.2f} MB)")
                return True
            else:
                print(f"❌ 输出文件未生成: {output_path}")
                return False
        else:
            print(f"❌ 推理失败，返回码: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ 推理超时（超过 300 秒）")
        return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False


def main():
    """主测试函数。"""
    print("=" * 80)
    print("Semantic-SAM CLI 测试套件")
    print("=" * 80)

    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n📁 输出目录: {OUTPUT_DIR}\n")

    # 运行所有测试
    results = []
    for input_image, output_name, level, model_size in TEST_CASES:
        success = run_test(input_image, output_name, level, model_size)
        results.append((input_image, output_name, success))

    # 总结
    print(f"\n{'=' * 80}")
    print("📊 测试总结")
    print(f"{'=' * 80}\n")

    passed = sum(1 for _, _, success in results if success)
    total = len(results)

    for input_image, output_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status}  {input_image:30} → {output_name}")

    print(f"\n总体: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败。")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
