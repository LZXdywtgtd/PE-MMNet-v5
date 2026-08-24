#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PE-MMNet v4 检查点系统验证脚本

用法：
  python verify_checkpoints.py                    # 检查所有变体
  python verify_checkpoints.py --variant resnet18  # 检查指定变体
  python verify_checkpoints.py --detail            # 显示完整元数据
"""

import os
import sys
import torch
import glob
from datetime import datetime
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def check_variant(variant: str, detail: bool = False) -> dict:
    """检查单个变体的检查点状态"""
    results = {}
    for ctype in ["best", "last"]:
        # 用 rglob 找所有匹配文件
        files = list(Path(CHECKPOINT_DIR).rglob(f"*_{ctype}.pt"))
        files = [f for f in files if variant in f.parts and "backup" not in f.parts]

        if not files:
            results[ctype] = None
            continue

        # 取最新的
        latest = max(files, key=lambda p: p.stat().st_mtime)
        try:
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            results[ctype] = {
                "path": str(latest.relative_to(CHECKPOINT_DIR.parent)),
                "epoch": ckpt.get("epoch", 0),
                "is_complete": ckpt.get("is_complete", False),
                "is_retrain_done": ckpt.get("is_retrain_done", False),
                "save_reason": ckpt.get("save_reason", "?"),
                "task_id": ckpt.get("task_id", "?"),
                "timestamp": ckpt.get("timestamp", "?"),
                "mtime": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%m-%d %H:%M:%S"),
            }
            if detail:
                print(f"\n  [{variant}/{ctype}] {latest.name}")
                for k, v in results[ctype].items():
                    print(f"    {k}: {v}")
        except Exception as e:
            results[ctype] = {"error": str(e)}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default=None, help="检查指定变体")
    parser.add_argument("--detail", action="store_true", help="显示完整元数据")
    args = parser.parse_args()

    variants = [args.variant] if args.variant else [
        "resnet18", "swin_yolo", "vit_yolo", "detr", "swin_yolo_patchtst"
    ]

    print(f"{'='*70}")
    print(f"  PE-MMNet v4 Checkpoint Verification Report")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    all_ok = True
    for variant in variants:
        ckpts = check_variant(variant, detail=args.detail)

        best = ckpts.get("best", {})
        last = ckpts.get("last", {})

        if best is None and last is None:
            print(f"\n  {variant:20s}  no checkpoint files")
            all_ok = False
            continue

        # 综合判定
        ep = max(
            best.get("epoch", 0) if best else 0,
            last.get("epoch", 0) if last else 0
        )
        complete = (best.get("is_complete") if best else False) or (last.get("is_complete") if last else False)
        retrain = (best.get("is_retrain_done") if best else False) or (last.get("is_retrain_done") if last else False)
        done = complete or retrain

        icon = "[OK]" if done else "[!!]"
        status = "complete" if done else ("interrupted" if ep > 0 else "no record")
        mtime = (best.get("mtime") if best else last.get("mtime")) if last else ""

        print(f"\n  {icon} {variant:20s}  {status}  epoch={ep}  {mtime}")

        if done and not args.detail:
            print(f"    best: {best.get('epoch', 0) if best else 0}  "
                  f"last: {last.get('epoch', 0) if last else 0}")
        if not done and ep > 0:
            print(f"    --> run training again to complete is_complete")

        if not done:
            all_ok = False

    print(f"\n{'='*70}")
    if all_ok:
        print(f"  [OK] All checkpoints OK")
    else:
        print(f"  [!!] Abnormal checkpoints found (epoch>0 but is_complete=False)")
        print(f"  Hint: python run_train.py --variant <variant> --epochs 2")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
