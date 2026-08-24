"""
等值线去除效果测试脚本

功能：
- 读取参数化扫描4的温度场图像
- 应用 ContourRemover 进行等值线去除
- 生成处理前后的对比图，方便评估效果

用法：
    python tools/test_contour_remover.py
    python tools/test_contour_remover.py --batch 参数化扫描4 --count 5
    python tools/test_contour_remover.py --batch 参数化扫描4 --count 3 --output output/contour_test
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.image_preprocessor import ContourRemover


def imread_unicode(filepath):
    """解决 OpenCV 中文路径读取问题"""
    try:
        # 使用 np.fromfile + cv2.imdecode
        with open(filepath, 'rb') as f:
            data = f.read()
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(filepath, img):
    """解决 OpenCV 中文路径写入问题"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        # 使用 cv2.imencode + 写入
        ext = os.path.splitext(filepath)[1]
        result, encoded = cv2.imencode(ext, img)
        if result:
            with open(filepath, 'wb') as f:
                f.write(encoded.tobytes())
            return True
        return False
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description='等值线去除效果测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python tools/test_contour_remover.py
  python tools/test_contour_remover.py --batch 参数化扫描4 --count 5
  python tools/test_contour_remover.py --batch 参数化扫描4 --count 3 --output output/contour_test
        """
    )
    parser.add_argument(
        '--batch', default='参数化扫描4',
        help='批次名称 (默认: 参数化扫描4)'
    )
    parser.add_argument(
        '--count', type=int, default=5,
        help='测试图像数量 (默认: 5)'
    )
    parser.add_argument(
        '--output', default='output/contour_test',
        help='输出目录 (默认: output/contour_test)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 数据路径
    DATA_ROOT = r"D:\Desktop\team_project\simulation\参考输入"
    batch_path = os.path.join(DATA_ROOT, args.batch)
    img_dir = os.path.join(batch_path, '温度')

    if not os.path.exists(img_dir):
        print(f"错误: 目录不存在 {img_dir}")
        print(f"请检查批次名称是否正确: {args.batch}")
        return

    # 获取图像列表
    images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    if not images:
        print(f"错误: 目录中没有 PNG 文件 {img_dir}")
        return

    # 限制数量
    images = images[:args.count]

    # 创建 ContourRemover
    remover = ContourRemover()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  等值线去除效果测试")
    print("=" * 60)
    print(f"\n批次: {args.batch}")
    print(f"图像目录: {img_dir}")
    print(f"测试数量: {len(images)} 张")
    print(f"输出目录: {args.output}")
    print()

    # 处理图像
    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(img_dir, img_name)

        # 读取图像（支持中文路径）
        img = imread_unicode(img_path)
        if img is None:
            print(f"  [{i}/{len(images)}] 跳过 (读取失败): {img_name}")
            continue

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 处理
        processed = remover(img_rgb)

        # RGB -> BGR (用于保存)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        # 创建对比图（左右拼接）
        # 调整尺寸使其相同
        original_resized = cv2.resize(img, (256, 256))

        # 在原始图上添加标签
        h, w = original_resized.shape[:2]
        cv2.putText(original_resized, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_bgr, "Processed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 拼接
        combined = np.hstack([original_resized, processed_bgr])

        # 保存（支持中文路径）
        output_path = os.path.join(args.output, f"compare_{img_name}")
        imwrite_unicode(output_path, combined)

        # 保存单独的处理后图像
        single_output = os.path.join(args.output, f"processed_{img_name}")
        imwrite_unicode(single_output, processed_bgr)

        print(f"  [{i}/{len(images)}] 已处理: {img_name}")

    print()
    print("=" * 60)
    print("  处理完成!")
    print("=" * 60)
    print(f"\n对比图: {args.output}/compare_*.png")
    print(f"处理后: {args.output}/processed_*.png")
    print()
    print("请检查对比效果：")
    print("  - 左侧 (Original): 原始图像，包含黑色等值线")
    print("  - 右侧 (Processed): 处理后，等值线已被去除")
    print()
    print("如果处理效果不理想，可调整 ContourRemover 参数：")
    print("  - black_threshold: 黑色阈值 (默认30)")
    print("  - inpaint_radius: 修复半径 (默认3)")
    print()


if __name__ == "__main__":
    main()
