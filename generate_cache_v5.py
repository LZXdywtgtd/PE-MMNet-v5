"""
v5 仿真缓存生成器（逐样本断点续跑 + 多进程并行）

用法（默认曲线 128px 1000 样本，约 4~5 小时，可随时中断续跑）:
    python generate_cache_v5.py --patch_size 128 --n_samples 1000 \
        --crack_stress_threshold_MPa 10 --workers 8

用法（拼装成训练侧可直接命中的单文件 npz）:
    python generate_cache_v5.py --patch_size 128 --n_samples 1000 \
        --crack_stress_threshold_MPa 10 --assemble_only

用法（验证缓存样本与现算逐位一致）:
    python generate_cache_v5.py --patch_size 64 --n_samples 2 \
        --crack_stress_threshold_MPa 10 --verify

设计（2026-09-29，断点续跑规范）：
- 每样本一个 chunk 文件 <chunk_dir>/s{idx:05d}.npz，原子写（.tmp→rename）
- 已存在的 chunk 直接跳过 → 中断后重跑只补缺
- chunk_dir 命名含全部影响输出的参数（patch/seed/thermal_hash/threshold）
- 全部完成后拼装为 patch_dataset_v5._cache_filename 命名的单文件 npz，
  训练侧 PatchDatasetV5(cache_dir=...) 直接命中，零仿真零拼装
- 训练 150ep 每样本约 37 分钟有 epoch 级 checkpoint（save_every=1），
  本脚本补齐生成侧 37 小时成本段的断点保险
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import hashlib
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEFAULT_PROFILE = {}  # 空 dict = PatchDatasetV5 默认烧成曲线（11.17h 物理口径）


def thermal_profile_hash(thermal_profile: dict) -> str:
    keys = sorted(thermal_profile.keys())
    payload = "|".join(f"{k}={thermal_profile[k]}" for k in keys)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def chunk_dir_name(args) -> str:
    th = thermal_profile_hash(args.thermal_profile)
    return (
        f"chunks_p{args.patch_size}_s{args.seed}"
        f"_t{th}_th{args.crack_stress_threshold_MPa}"
    )


def generate_one(idx: int, cfg: dict) -> str:
    """生成单个样本并原子写 chunk。返回状态字符串。"""
    out_path = os.path.join(cfg["chunk_dir"], f"s{idx:05d}.npz")
    if os.path.exists(out_path):
        return f"s{idx:05d} skip"
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=cfg["chunk_dir"], prefix=f".s{idx:05d}.", suffix=".tmp"
    )
    os.close(tmp_fd)
    try:
        from data.patch_dataset_v5 import PatchDatasetV5

        ds = PatchDatasetV5(
            patch_size=cfg["patch_size"],
            n_samples=1,  # 只为借用 _generate_sample；不影响输出
            seed=cfg["seed"] + idx,  # _generate_sample 内部用 seed+idx
            crack_stress_threshold_MPa=cfg["threshold"],
            thermal_profile=cfg["thermal_profile"] or None,
            precompute=False,
        )
        sample = ds._generate_sample(0)
        # 复刻 _save_to_npz 的单样本打包（padding/mask 逻辑一致）
        k = sample["true_keypoints"].size(0)
        m = sample["true_crack_pixels"].size(0)
        kpts = np.full((1, max(k, 1), 2), -1.0, dtype=np.float32)
        kpt_mask = np.zeros((1, max(k, 1)), dtype=bool)
        if k > 0:
            kpts[0, :k] = sample["true_keypoints"].numpy()
            kpt_mask[0, :k] = True
        px = np.full((1, max(m, 1), 2), -1.0, dtype=np.float32)
        px_mask = np.zeros((1, max(m, 1)), dtype=bool)
        if m > 0:
            px[0, :m] = sample["true_crack_pixels"].numpy()
            px_mask[0, :m] = True
        # np.savez* 收到字符串路径时若后缀非 .npz 会自动追加 .npz，
        # 所以必须传文件句柄（.tmp 后缀 + 原子 rename 需要路径不变）
        with open(tmp_path, "wb") as f:
            np.savez_compressed(
                f,
                x_1d=sample["x_1d"].numpy()[None],
                x_2d=sample["x_2d"].numpy()[None],
                true_bbox=sample["true_bbox"].numpy()[None],
                true_keypoints=kpts,
                keypoint_mask=kpt_mask,
                true_crack_pixels=px,
                pixel_mask=px_mask,
            )
        # Windows: os.replace 是原子操作（同卷）
        os.replace(tmp_path, out_path)
        meta = sample.get("metadata", {})
        return (
            f"s{idx:05d} done "
            f"peak={meta.get('peak_stress_MPa', float('nan')):.2f}MPa "
            f"kpts={k}"
        )
    finally:
        import glob
        for leftover in glob.glob(tmp_path + "*"):
            os.remove(leftover)


def assemble(args, chunk_dir: str) -> str:
    """把全部 chunk 拼装为训练侧 _cache_filename 命名的单文件 npz。"""
    from data.patch_dataset_v5 import _cache_filename

    th = thermal_profile_hash(args.thermal_profile)
    cache_name = _cache_filename(
        args.patch_size, args.n_samples, args.seed,
        th, args.crack_stress_threshold_MPa,
    )
    out_path = os.path.join(args.cache_dir, cache_name)
    if os.path.exists(out_path):
        print(f"[assemble] 已存在，跳过: {out_path}")
        return out_path

    t0 = time.time()
    xs1, xs2, bboxes, kpts_list, px_list = [], [], [], [], []
    for idx in range(args.n_samples):
        p = os.path.join(chunk_dir, f"s{idx:05d}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺 chunk: {p}（先用生成模式补齐）")
        d = np.load(p)
        xs1.append(d["x_1d"][0])
        xs2.append(d["x_2d"][0])
        bboxes.append(d["true_bbox"][0])
        kpt_mask = d["keypoint_mask"][0]
        px_mask = d["pixel_mask"][0]
        kpts_list.append(d["true_keypoints"][0][kpt_mask])
        px_list.append(d["true_crack_pixels"][0][px_mask])

    n = args.n_samples
    # 变长 → padding + mask（复刻 _save_to_npz 逻辑）
    max_K = max(k.shape[0] for k in kpts_list)
    max_N = max(m.shape[0] for m in px_list)
    true_kpts = np.full((n, max_K, 2), -1.0, dtype=np.float32)
    kpt_mask = np.zeros((n, max_K), dtype=bool)
    true_px = np.full((n, max_N, 2), -1.0, dtype=np.float32)
    px_mask = np.zeros((n, max_N), dtype=bool)
    for i, (k, m) in enumerate(zip(kpts_list, px_list)):
        true_kpts[i, : k.shape[0]] = k
        kpt_mask[i, : k.shape[0]] = True
        true_px[i, : m.shape[0]] = m
        px_mask[i, : m.shape[0]] = True

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=args.cache_dir, prefix=".assemble.", suffix=".tmp"
    )
    os.close(tmp_fd)
    try:
        # np.savez* 收到字符串路径时若后缀非 .npz 会自动追加 .npz，
        # 所以必须传文件句柄（.tmp 后缀 + 原子 rename 需要路径不变）
        with open(tmp_path, "wb") as f:
            np.savez_compressed(
                f,
                x_1d=np.stack(xs1), x_2d=np.stack(xs2),
                true_bbox=np.stack(bboxes),
                true_keypoints=true_kpts, keypoint_mask=kpt_mask,
                true_crack_pixels=true_px, pixel_mask=px_mask,
            )
        os.replace(tmp_path, out_path)
    finally:
        import glob
        for leftover in glob.glob(tmp_path + "*"):
            os.remove(leftover)

    size_mb = os.path.getsize(out_path) / 1e6
    print(
        f"[assemble] {out_path} ({size_mb:.0f} MB, "
        f"{time.time() - t0:.0f}s)"
    )
    return out_path


def verify(args, chunk_dir: str) -> bool:
    """缓存样本 vs 现算样本逐位对比。"""
    ok = True
    for idx in [0, args.n_samples // 2, args.n_samples - 1]:
        p = os.path.join(chunk_dir, f"s{idx:05d}.npz")
        d = np.load(p)
        cached_x2d = d["x_2d"][0]

        from data.patch_dataset_v5 import PatchDatasetV5

        ds = PatchDatasetV5(
            patch_size=args.patch_size,
            n_samples=1,
            seed=args.seed + idx,
            crack_stress_threshold_MPa=args.crack_stress_threshold_MPa,
            thermal_profile=args.thermal_profile or None,
            precompute=False,
        )
        fresh = ds._generate_sample(0)
        if not np.array_equal(cached_x2d, fresh["x_2d"].numpy()):
            diff = np.abs(cached_x2d - fresh["x_2d"].numpy()).max()
            print(f"  [FAIL] s{idx:05d}: max|Δx_2d| = {diff}")
            ok = False
        else:
            print(f"  [OK] s{idx:05d}: x_2d 逐位一致")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="v5 仿真缓存生成（断点续跑 + 多进程）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crack_stress_threshold_MPa", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--fast_thermal", action="store_true",
        help="1min 短周期（管线测试用）",
    )
    ap.add_argument(
        "--cache_dir", default="logs/sim_cache",
        help="npz 缓存根目录（chunk 与拼装产物都在这）",
    )
    ap.add_argument(
        "--assemble_only", action="store_true",
        help="跳过生成，只做 chunk→单文件拼装",
    )
    ap.add_argument(
        "--verify", action="store_true",
        help="抽查 3 个样本与现算逐位对比",
    )
    args = ap.parse_args()

    args.thermal_profile = (
        {
            "ramp_up_c_per_min": 1260.0,
            "soak_temp_c": 1280.0,
            "soak_duration_min": 1.0,
            "cool_down_c_per_min": 1260.0,
        }
        if args.fast_thermal
        else DEFAULT_PROFILE
    )

    os.makedirs(args.cache_dir, exist_ok=True)
    cdir = os.path.join(args.cache_dir, chunk_dir_name(args))
    os.makedirs(cdir, exist_ok=True)
    print(f"[config] chunk_dir = {cdir}")
    print(
        f"[config] p{args.patch_size} n={args.n_samples} "
        f"seed={args.seed} th={args.crack_stress_threshold_MPa} "
        f"workers={args.workers} "
        f"thermal={'fast' if args.fast_thermal else 'default(11.17h)'}"
    )

    if args.assemble_only:
        assemble(args, cdir)
        return

    if args.verify:
        done = [f for f in os.listdir(cdir) if f.endswith(".npz")]
        print(f"[verify] 已有 chunk {len(done)}/{args.n_samples}")
        sys.exit(0 if verify(args, cdir) else 1)

    # ── 生成主循环（断点续跑：已存在的 chunk 跳过）──
    pending = [
        i for i in range(args.n_samples)
        if not os.path.exists(os.path.join(cdir, f"s{i:05d}.npz"))
    ]
    t0 = time.time()
    print(f"[start] 待生成 {len(pending)}/{args.n_samples}")
    if pending:
        done_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_one, i, {
                    "chunk_dir": cdir,
                    "patch_size": args.patch_size,
                    "seed": args.seed,
                    "threshold": args.crack_stress_threshold_MPa,
                    "thermal_profile": args.thermal_profile,
                }): i
                for i in pending
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    msg = fut.result()
                    done_count += 1
                    elapsed = time.time() - t0
                    rate = done_count / max(elapsed, 1e-9)
                    eta = (len(pending) - done_count) / max(rate, 1e-9) / 60
                    print(
                        f"[{done_count}/{len(pending)}] {msg}  "
                        f"| {rate:.2f} 样本/s, ETA {eta:.0f} min"
                    )
                except Exception as e:
                    print(f"[ERROR] s{idx:05d}: {e!r}")
                    # 不中断：chunk 缺失会在 assemble 时报 FileNotFoundError

    n_done = sum(
        1 for i in range(args.n_samples)
        if os.path.exists(os.path.join(cdir, f"s{i:05d}.npz"))
    )
    dt = time.time() - t0
    print(f"[done] {n_done}/{args.n_samples} chunk 就绪，耗时 {dt/60:.1f} min")

    # ── 拼装（全部就绪才执行）──
    if n_done == args.n_samples:
        assemble(args, cdir)
    else:
        print(
            f"[WARN] 缺 {args.n_samples - n_done} 个 chunk，"
            f"未拼装。重跑本命令可续。"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
