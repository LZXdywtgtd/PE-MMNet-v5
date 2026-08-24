"""
patch 仿真器结果可视化（V5A1 — 可视化辅助工具）

把 PatchSimulator.simulate() 返回的结果画成 4 联图：
- 左上：最终温度场
- 右上：最终应力场
- 左下：裂纹 mask + 关键点 + bbox
- 右下：温度时序曲线（中心点）

用法：
    python tools/patch_simulator_visualizer.py --simulate
    或：
    from tools.patch_simulator_visualizer import visualize_result
    visualize_result(result, save_path="output/v5_sim_sample.png")

详见：
- tasks/team_v5_alpha.json 中 V5A1 deliverable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_result(
    result: dict,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    绘制 4 联可视化图

    Args:
        result:    PatchSimulator.simulate() 的返回值
        save_path: 保存路径（None=不保存）
        show:      是否 plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f"PE-MMNet v5 Patch Simulation\n"
        f"Patch={result['metadata']['patch_size']}×{result['metadata']['patch_size']}, "
        f"Peak σ={result['metadata']['peak_stress_MPa']:.1f} MPa, "
        f"K={result['metadata']['n_keypoints']} keypoints",
        fontsize=13,
    )

    # === 左上：最终温度场 ===
    ax = axes[0, 0]
    T_final = result["temperature_field"][-1]
    im = ax.imshow(T_final, cmap="inferno", origin="upper")
    ax.set_title("Final Temperature Field (°C)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # === 右上：最终应力场 ===
    ax = axes[0, 1]
    sigma_final = result["stress_field"][-1]
    im = ax.imshow(sigma_final / 1e6, cmap="RdBu_r", origin="upper",
                   vmin=-sigma_final.max()/1e6, vmax=sigma_final.max()/1e6)
    ax.set_title("Final Stress Field (MPa, blue=comp / red=tensile)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # === 左下：裂纹 mask + 关键点 + bbox ===
    ax = axes[1, 0]
    mask = result["crack_mask"]
    ax.imshow(mask, cmap="gray_r", origin="upper", alpha=0.5)

    # 画关键点
    kpts = result["crack_keypoints"]
    if len(kpts) > 0:
        ax.scatter(kpts[:, 1], kpts[:, 0],
                   c="red", s=40, marker="o", edgecolors="white",
                   linewidths=0.8, label=f"Keypoints (K={len(kpts)})")
        # 画关键点连线（顺序）
        ax.plot(kpts[:, 1], kpts[:, 0], "r-", alpha=0.6, linewidth=1.2)

    # 画 bbox
    bbox = result["crack_bbox"]
    H = W = result["metadata"]["patch_size"]
    x1, y1, x2, y2 = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
    rect = Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="lime", facecolor="none", linestyle="--",
        label="BBox",
    )
    ax.add_patch(rect)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("Crack Mask + Keypoints + BBox")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

    # === 右下：温度时序曲线（中心点 + 边角） ===
    ax = axes[1, 1]
    T_history = result["temperature_field"]  # (T_samples, H, W)
    H_p, W_p = T_history.shape[1], T_history.shape[2]
    sample_points = {
        "Center":    (H_p // 2, W_p // 2),
        "Top-Left":  (10, 10),
        "Top-Right": (10, W_p - 10),
        "Bot-Center":(H_p - 10, W_p // 2),
    }
    time_s = np.linspace(0, result["metadata"]["total_time_h"] * 3600,
                        T_history.shape[0])
    for name, (yy, xx) in sample_points.items():
        ax.plot(time_s / 3600, T_history[:, yy, xx], label=name, linewidth=1.5)

    # 标记阶段
    total_h = result["metadata"]["total_time_h"]
    ax.axvline(x=4.0, color="orange", linestyle=":", alpha=0.5, label="End of ramp-up")
    ax.axvline(x=4.5, color="red", linestyle=":", alpha=0.5, label="End of soak")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Profile at Sample Points")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[PatchVisualizer] 保存到 {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_stress_evolution(result: dict, save_path: str | None = None) -> None:
    """
    应力随时间的演化（峰值 + 中心点）

    Args:
        result:    PatchSimulator.simulate() 的返回值
        save_path: 保存路径
    """
    sigma_history = result["stress_field"]  # (T_samples, H, W)
    peak_stress = sigma_history.max(axis=(1, 2)) / 1e6  # MPa
    center_stress = sigma_history[:, sigma_history.shape[1] // 2,
                                   sigma_history.shape[2] // 2] / 1e6

    time_s = np.linspace(0, result["metadata"]["total_time_h"] * 3600,
                        len(peak_stress))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_s / 3600, peak_stress, label="Peak stress (max)", linewidth=2)
    ax.plot(time_s / 3600, center_stress, label="Center stress", linewidth=1.5, alpha=0.7)
    ax.axhline(y=result["metadata"]["peak_stress_MPa"], color="red",
               linestyle="--", alpha=0.5, label="Final peak")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Stress Evolution During Firing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[StressEvolution] 保存到 {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v5 patch 仿真可视化")
    parser.add_argument("--simulate", action="store_true",
                        help="先跑一次仿真再画图")
    parser.add_argument("--output", type=str,
                        default="output/v5_sim_sample.png",
                        help="输出 PNG 路径")
    args = parser.parse_args()

    if args.simulate:
        from data.patch_simulator_v5 import PatchSimulator

        print("[Visualizer] 跑一次冒烟仿真…")
        sim = PatchSimulator(
            patch_size=64,
            physical_size_cm=2.0,
            thermal_profile={
                "ramp_up_c_per_min": 50.0,
                "soak_temp_c": 200.0,
                "soak_duration_min": 1.0,
                "cool_down_c_per_min": 30.0,
            },
            crack_stress_threshold_MPa=10.0,
        )
        result = sim.simulate(sample_interval_steps=200, verbose=True)

        visualize_result(result, save_path=args.output)
        visualize_stress_evolution(
            result, save_path=str(Path(args.output).with_name("v5_stress_evolution.png"))
        )
    else:
        print("用法：python tools/patch_simulator_visualizer.py --simulate")