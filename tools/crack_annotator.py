"""
裂纹轨迹标注工具（V5B3：v5-β 实现，纯 Python CLI）

> 继承关系：本工具继承自 v5-α 占位（V5A5 之后补充）。
> v5-α 无真实烧后图，本工具无输入数据；
> v5-β 启动后接入陶艺作坊真实烧后图时启用。

设计：
- 提供两种模式：
  1. **GUI 模式（推荐）**：基于 matplotlib + 鼠标点击的可视化标注
     - 依赖：matplotlib（已装） + Pillow（已装）
     - 功能：加载图 → 自动骨架提取 → 手动调整关键点 → 保存 JSON
  2. **CLI 批处理模式**：从文件夹批量提取 → 自动骨架 → 输出 JSON
     - 不依赖 GUI，适合服务器批处理

为什么不用 streamlit？
- 当前 pe_mmnet 环境未安装 streamlit
- matplotlib + 鼠标交互足够覆盖标注需求
- v5-β 启动后若需 Web GUI，可用 streamlit 包装本模块的核心函数

输出格式（JSON）：
    {
        "image_path": "...",
        "ceramic_id": "...",
        "timestamp": "...",
        "image_size": [H, W],
        "crack_bbox": [x1, y1, x2, y2],   # 像素坐标
        "crack_keypoints": [[y1, x1], ...],  # 像素坐标
        "skeleton_method": "canny_skeletonize" | "manual",
        "metadata": {...},
    }

详见：
- docs/v5_范围说明.md §6 标注流程
- docs/v5_架构设计.md §8.3 v5-β 标注
- tasks/team_v5_alpha.json 中 V5B3
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# 可选依赖（GUI 模式需要 matplotlib；CLI 模式只依赖 numpy）
try:
    import matplotlib
    matplotlib.use("TkAgg")  # Windows 默认；Linux 可改为 'Qt5Agg'
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from skimage import io as skio
    from skimage import feature, morphology
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ============================================================
#  核心：自动骨架提取
# ============================================================

def auto_extract_crack(
    image: np.ndarray,
    min_keypoints: int = 8,
    max_keypoints: int = 16,
    canny_sigma: float = 2.0,
    verbose: bool = False,
) -> dict:
    """
    自动从烧后图提取裂纹轨迹

    流程：
        1. Canny 边缘检测
        2. 形态学闭运算填补断裂
        3. skeletonize 骨架化
        4. 沿骨架均匀采样 N 个关键点（动态 N ∈ [min, max]）
        5. bbox = 骨架外接矩形

    Args:
        image: H×W×3 uint8 RGB 图像
        min_keypoints / max_keypoints: 关键点数范围
        canny_sigma: Canny 高斯 sigma
        verbose: 是否打印

    Returns:
        dict: {
            "keypoints": np.ndarray (K, 2) (y, x),
            "bbox": (x1, y1, x2, y2),
            "skeleton": np.ndarray (H, W) bool,
        }

    注：v5-α 当前无真实图像测试；v5-β 启动后用陶艺作坊数据验证。
    """
    if not _HAS_SKIMAGE:
        raise ImportError(
            "auto_extract_crack 需要 scikit-image；当前 pe_mmnet 环境未装。"
        )

    # 转灰度
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    # 1. Canny 边缘
    edges = feature.canny(gray, sigma=canny_sigma)

    # 2. 形态学闭运算（半径 2）
    closed = morphology.binary_closing(edges, morphology.disk(2))

    # 3. 骨架化
    skeleton = morphology.skeletonize(closed)

    # 4. 提取骨架像素坐标
    ys, xs = np.where(skeleton)
    if len(ys) == 0:
        if verbose:
            print(f"  [auto] 未检测到裂纹骨架")
        return {
            "keypoints": np.zeros((0, 2), dtype=np.float32),
            "bbox": (0, 0, 0, 0),
            "skeleton": skeleton,
        }

    # 沿骨架均匀采样 N 个点
    n_skeleton = len(ys)
    n_target = max(min_keypoints, min(max_keypoints, n_skeleton // 30))
    # 等弧长采样
    if n_target <= 2:
        idxs = np.linspace(0, n_skeleton - 1, max(n_target, 2)).astype(int)
    else:
        idxs = np.linspace(0, n_skeleton - 1, n_target).astype(int)
    keypoints = np.stack([ys[idxs], xs[idxs]], axis=1).astype(np.float32)

    # 5. bbox（外接矩形，含少量 padding）
    pad = 5
    bbox = (
        float(xs.min() - pad), float(ys.min() - pad),
        float(xs.max() + pad), float(ys.max() + pad),
    )

    if verbose:
        print(f"  [auto] 骨架像素={n_skeleton}, 关键点={len(keypoints)}, "
              f"bbox={bbox}")
    return {
        "keypoints": keypoints,
        "bbox": bbox,
        "skeleton": skeleton,
    }


# ============================================================
#  核心：保存标注
# ============================================================

def save_annotation(
    image_path: str,
    keypoints: np.ndarray,
    bbox: tuple,
    ceramic_id: str = "",
    metadata: dict | None = None,
    output_path: str | None = None,
    skeleton_method: str = "manual",
) -> str:
    """
    保存标注到 JSON

    Returns:
        实际写入的文件路径
    """
    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_annotation.json"

    # 自动推断 ceramic_id
    if not ceramic_id:
        stem = Path(image_path).stem
        # 约定文件名包含 cartXX_layerY_... 或陶瓷ID
        parts = stem.split("_")
        for p in parts:
            if p.startswith("cart"):
                ceramic_id = p
                break
        if not ceramic_id:
            ceramic_id = stem

    ann = {
        "image_path": str(image_path),
        "ceramic_id": ceramic_id,
        "timestamp": datetime.now().isoformat(),
        "image_size": None,  # 由 GUI 模式填充
        "crack_bbox": list(bbox),
        "crack_keypoints": keypoints.tolist(),
        "skeleton_method": skeleton_method,
        "metadata": metadata or {},
    }

    if _HAS_PIL and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                ann["image_size"] = list(img.size[::-1])  # (H, W)
        except Exception:
            pass

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    return output_path


# ============================================================
#  核心：加载标注
# ============================================================

def load_annotation(path: str) -> dict:
    """从 JSON 加载标注"""
    with open(path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    ann["crack_keypoints"] = np.array(ann["crack_keypoints"], dtype=np.float32)
    return ann


# ============================================================
#  GUI 标注器（matplotlib 鼠标交互）
# ============================================================

class CrackAnnotatorGUI:
    """
    基于 matplotlib 的交互式裂纹标注器

    用法：
        annotator = CrackAnnotatorGUI(image_path="img.png")
        annotator.run()

    操作：
        - 左键：添加关键点
        - 右键：删除最近的关键点
        - 中键拖动：移动最近的关键点
        - 'a'：自动骨架提取
        - 'r'：清空所有关键点
        - 's'：保存标注
        - 'q'：退出
    """

    def __init__(
        self,
        image_path: str,
        output_path: str | None = None,
        ceramic_id: str = "",
        min_keypoints: int = 8,
        max_keypoints: int = 16,
    ):
        if not _HAS_MPL:
            raise ImportError("GUI 模式需要 matplotlib")
        if not _HAS_PIL:
            raise ImportError("GUI 模式需要 Pillow")

        self.image_path = image_path
        self.output_path = output_path
        self.ceramic_id = ceramic_id
        self.min_keypoints = min_keypoints
        self.max_keypoints = max_keypoints

        # 加载图像
        img_pil = Image.open(image_path).convert("RGB")
        self.image = np.array(img_pil)
        self.H, self.W = self.image.shape[:2]

        # 标注状态
        self.keypoints: list[tuple[float, float]] = []  # (y, x)
        self.bbox: tuple = (0, 0, 0, 0)
        self.skeleton: np.ndarray | None = None
        self.drag_idx: int | None = None

        # matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self._setup_plot()
        self._connect_events()

    def _setup_plot(self):
        """初始化图像 + 关键点 + bbox 显示"""
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"裂纹标注器 - {Path(self.image_path).name}\n"
            f"操作: 左键=添加 | 右键=删除 | 'a'=自动 | 'r'=清空 | 's'=保存 | 'q'=退出",
            fontsize=11,
        )
        self.scatter = self.ax.scatter([], [], c="red", s=80, marker="o",
                                       edgecolors="yellow", linewidths=1.5,
                                       zorder=5)
        self.bbox_rect = None

    def _connect_events(self):
        """绑定鼠标/键盘事件"""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        y, x = float(event.ydata), float(event.xdata)
        if event.button == 1:  # 左键：添加
            if len(self.keypoints) < self.max_keypoints:
                self.keypoints.append((y, x))
                self._refresh()
                print(f"  [+ ] ({y:.1f}, {x:.1f}) 共 {len(self.keypoints)} 点")
        elif event.button == 3:  # 右键：删除最近
            if self.keypoints:
                # 找最近的关键点
                dists = [(k[0] - y) ** 2 + (k[1] - x) ** 2 for k in self.keypoints]
                idx = int(np.argmin(dists))
                removed = self.keypoints.pop(idx)
                self._refresh()
                print(f"  [- ] ({removed[0]:.1f}, {removed[1]:.1f}) "
                      f"剩余 {len(self.keypoints)} 点")

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        # 中键拖动：找最近关键点并跟随
        if self.keypoints:
            y, x = float(event.ydata), float(event.xdata)
            dists = [(k[0] - y) ** 2 + (k[1] - x) ** 2 for k in self.keypoints]
            idx = int(np.argmin(dists))
            if dists[idx] < 50 ** 2:
                self.drag_idx = idx
                self.keypoints[idx] = (y, x)
                self._refresh()

    def _on_release(self, event):
        self.drag_idx = None

    def _on_key(self, event):
        k = event.key.lower()
        if k == "a":
            self._auto_extract()
        elif k == "r":
            self.keypoints.clear()
            self.skeleton = None
            self._refresh()
            print(f"  [清空] 共 0 点")
        elif k == "s":
            self._save()
        elif k == "q":
            plt.close(self.fig)
            print(f"  [退出]")

    def _auto_extract(self):
        """自动提取（GUI 内部用）"""
        try:
            result = auto_extract_crack(
                self.image,
                min_keypoints=self.min_keypoints,
                max_keypoints=self.max_keypoints,
                verbose=True,
            )
            self.keypoints = [(float(k[0]), float(k[1]))
                              for k in result["keypoints"]]
            self.bbox = result["bbox"]
            self.skeleton = result["skeleton"]
            self._refresh()
            print(f"  [auto] 提取 {len(self.keypoints)} 个关键点 + bbox")
        except Exception as e:
            print(f"  [auto 失败] {e}")

    def _refresh(self):
        """刷新散点 + bbox 显示"""
        if self.keypoints:
            ys = [k[0] for k in self.keypoints]
            xs = [k[1] for k in self.keypoints]
            self.scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))

        # bbox 矩形
        if self.bbox_rect is not None:
            self.bbox_rect.remove()
        if self.keypoints and len(self.keypoints) >= 2:
            ys = [k[0] for k in self.keypoints]
            xs = [k[1] for k in self.keypoints]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            self.bbox_rect = self.ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="cyan", linewidth=2, zorder=4,
                )
            )
        self.fig.canvas.draw_idle()

    def _save(self):
        """保存当前标注"""
        if not self.keypoints:
            print(f"  [警告] 无关键点，未保存")
            return
        kpts_arr = np.array(self.keypoints, dtype=np.float32)
        ys = [k[0] for k in self.keypoints]
        xs = [k[1] for k in self.keypoints]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        path = save_annotation(
            image_path=self.image_path,
            keypoints=kpts_arr,
            bbox=bbox,
            ceramic_id=self.ceramic_id,
            skeleton_method="manual" if self.skeleton is None else "canny_skeletonize",
            output_path=self.output_path,
        )
        print(f"  [保存] → {path}")

    def run(self):
        """启动 GUI（阻塞）"""
        print(f"\n[启动 GUI] 图像: {self.image_path}")
        print(f"  尺寸: {self.H}×{self.W}")
        print(f"  最小关键点: {self.min_keypoints}, 最大: {self.max_keypoints}")
        plt.show()


# ============================================================
#  CLI 批处理模式
# ============================================================

def batch_auto_extract(
    input_dir: str,
    output_dir: str | None = None,
    min_keypoints: int = 8,
    max_keypoints: int = 16,
    pattern: str = "*.png",
    canny_sigma: float = 2.0,
    verbose: bool = True,
) -> list[str]:
    """
    CLI 批处理：从文件夹批量提取骨架 → JSON

    Args:
        input_dir: 输入图像文件夹
        output_dir: 输出 JSON 文件夹（None → 与输入同目录）
        min_keypoints / max_keypoints: 关键点数范围
        pattern: 文件名匹配（默认 png）
        canny_sigma: Canny sigma
        verbose: 是否打印

    Returns:
        输出的 JSON 文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if output_dir is None:
        output_dir = str(input_path / "annotations")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(input_path.glob(pattern))
    if not files:
        print(f"  [批处理] 无匹配文件: {input_dir}/{pattern}")
        return []

    output_files = []
    for f in files:
        if verbose:
            print(f"  [处理] {f.name}")
        try:
            if _HAS_PIL:
                img_pil = Image.open(f).convert("RGB")
                img = np.array(img_pil)
            else:
                img = skio.imread(str(f))

            result = auto_extract_crack(
                img,
                min_keypoints=min_keypoints,
                max_keypoints=max_keypoints,
                canny_sigma=canny_sigma,
                verbose=False,
            )
            if len(result["keypoints"]) == 0:
                if verbose:
                    print(f"    [跳过] 未检测到裂纹")
                continue

            out_path = os.path.join(output_dir, f"{f.stem}_annotation.json")
            save_annotation(
                image_path=str(f),
                keypoints=result["keypoints"],
                bbox=result["bbox"],
                skeleton_method="canny_skeletonize",
                output_path=out_path,
            )
            output_files.append(out_path)
            if verbose:
                print(f"    [OK] {len(result['keypoints'])} 点 → {out_path}")
        except Exception as e:
            print(f"    [失败] {f.name}: {e}")

    print(f"\n  [完成] 处理 {len(files)} 张，输出 {len(output_files)} 个 JSON")
    return output_files


# ============================================================
#  CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="v5-β 裂纹标注工具（V5B3）",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # GUI 模式
    gui = sub.add_parser("gui", help="交互式标注（matplotlib）")
    gui.add_argument("--image", required=True, help="图像路径")
    gui.add_argument("--output", default=None, help="输出 JSON 路径")
    gui.add_argument("--ceramic-id", default="", help="陶瓷 ID")
    gui.add_argument("--min-kpts", type=int, default=8)
    gui.add_argument("--max-kpts", type=int, default=16)

    # CLI 批处理模式
    batch = sub.add_parser("batch", help="批量自动提取（无 GUI）")
    batch.add_argument("--input-dir", required=True, help="输入图文件夹")
    batch.add_argument("--output-dir", default=None, help="输出 JSON 文件夹")
    batch.add_argument("--min-kpts", type=int, default=8)
    batch.add_argument("--max-kpts", type=int, default=16)
    batch.add_argument("--pattern", default="*.png", help="文件名匹配")
    batch.add_argument("--canny-sigma", type=float, default=2.0)

    # 校验模式（看标注是否完整）
    val = sub.add_parser("validate", help="校验 JSON 标注完整性")
    val.add_argument("json_path", help="待校验的 JSON 文件")

    args = parser.parse_args()

    if args.mode == "gui":
        if not _HAS_MPL:
            print("  [错误] matplotlib 未装，无法启动 GUI")
            sys.exit(1)
        annotator = CrackAnnotatorGUI(
            image_path=args.image,
            output_path=args.output,
            ceramic_id=args.ceramic_id,
            min_keypoints=args.min_kpts,
            max_keypoints=args.max_kpts,
        )
        annotator.run()
    elif args.mode == "batch":
        batch_auto_extract(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_keypoints=args.min_kpts,
            max_keypoints=args.max_kpts,
            pattern=args.pattern,
            canny_sigma=args.canny_sigma,
        )
    elif args.mode == "validate":
        ann = load_annotation(args.json_path)
        n_kpts = len(ann["crack_keypoints"])
        bbox = ann["crack_bbox"]
        print(f"  [校验] {args.json_path}")
        print(f"    图像:   {ann.get('image_path')}")
        print(f"    陶瓷ID: {ann.get('ceramic_id')}")
        print(f"    尺寸:   {ann.get('image_size')}")
        print(f"    关键点: {n_kpts}")
        print(f"    bbox:   {bbox}")
        print(f"    方法:   {ann.get('skeleton_method')}")
        if n_kpts < 8 or n_kpts > 16:
            print(f"    ⚠️ 关键点数 {n_kpts} 超出 [8, 16]")
        else:
            print(f"    ✅ 通过")


# ============================================================
#  冒烟测试
# ============================================================

def _smoke_test():
    print("=" * 60)
    print("crack_annotator 冒烟测试（CLI 部分）")
    print("=" * 60)

    # 1) 自动骨架提取（合成测试图）
    print("\n[auto_extract_crack 合成测试]")
    if _HAS_SKIMAGE and _HAS_PIL:
        # 合成一张带裂纹的图（白色背景 + 黑线）
        H, W = 128, 128
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
        # 画一条 S 形裂纹
        for i in range(20):
            y = int(20 + i * 4)
            x = int(64 + 20 * np.sin(i * 0.5))
            if 0 <= y < H and 0 <= x < W:
                img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 0

        result = auto_extract_crack(img, min_keypoints=4, max_keypoints=8, verbose=True)
        print(f"  关键点: {result['keypoints'].shape}")
        print(f"  bbox:   {result['bbox']}")
        assert result["keypoints"].shape[1] == 2
        assert len(result["bbox"]) == 4
    else:
        print(f"  [跳过] skimage 或 PIL 未装")

    # 2) save / load annotation
    print("\n[save / load annotation]")
    test_kpts = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    test_bbox = (15, 5, 65, 55)
    out = save_annotation(
        image_path="test_image.png",  # 文件不存在也行，只测试 JSON
        keypoints=test_kpts,
        bbox=test_bbox,
        ceramic_id="test_c01",
        output_path="logs/_test_annotation.json",
    )
    print(f"  保存到: {out}")
    ann = load_annotation(out)
    print(f"  加载后: ceramic_id={ann['ceramic_id']}, K={len(ann['crack_keypoints'])}")
    assert ann["ceramic_id"] == "test_c01"
    assert np.allclose(ann["crack_keypoints"], test_kpts)
    print(f"  ✅ 序列化/反序列化一致")
    os.remove(out)

    # 3) CLI validate
    print("\n[CLI validate 模式]")
    test_kpts2 = np.array([[10, 20]] * 10, dtype=np.float32)
    test_bbox2 = (15, 5, 65, 55)
    out2 = save_annotation(
        image_path="test.png", keypoints=test_kpts2, bbox=test_bbox2,
        ceramic_id="test_c02", output_path="logs/_test_ann2.json",
    )
    sys.argv = ["crack_annotator.py", "validate", out2]
    main()
    os.remove(out2)

    print("\n[ALL PASS] ✅")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        _smoke_test()