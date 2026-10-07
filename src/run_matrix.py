"""实验矩阵批量执行入口。

用法示例
--------
    python -m src.run_matrix --list                    # 只列出实验清单
    python -m src.run_matrix --groups A1               # 只跑 A1 组
    python -m src.run_matrix --only a1_resnet18 c1_lr1e-4
    python -m src.run_matrix --all                     # 跑完整矩阵（自动跳过已完成）
    python -m src.run_matrix --all --force             # 强制重跑

已完成的实验（存在 metrics.json）默认会被跳过，因此中断后可直接重跑同一命令续跑。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from . import config as C
from .experiments import all_experiments, select
from .train import run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="猫的十二分类 —— 实验矩阵执行器")
    p.add_argument("--groups", nargs="+", default=None, help="只运行指定组，如 A1 C1")
    p.add_argument("--only", nargs="+", default=None, help="只运行指定 run_id")
    p.add_argument("--all", action="store_true", help="运行全部实验")
    p.add_argument("--list", action="store_true", help="列出实验清单后退出")
    p.add_argument("--force", action="store_true", help="已完成的实验也强制重跑")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数（用于快速验证）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfgs = all_experiments()

    if args.list:
        for c in cfgs:
            print(f"{c.group:4s} {c.run_id:28s} arch={c.arch:20s} pretrained={str(c.pretrained):5s} {c.note}")
        print(f"\n共 {len(cfgs)} 次实验")
        return 0

    if args.only:
        wanted = set(args.only)
        cfgs = [c for c in cfgs if c.run_id in wanted]
        missing = wanted - {c.run_id for c in cfgs}
        if missing:
            print(f"[warn] 未找到 run_id: {sorted(missing)}", file=sys.stderr)
    elif args.groups:
        cfgs = select(args.groups)
    elif not args.all:
        print("请指定 --all / --groups / --only，或用 --list 查看清单", file=sys.stderr)
        return 2

    if args.epochs is not None:
        for c in cfgs:
            c.epochs = args.epochs

    C.ensure_dirs()

    done, failed, skipped = [], [], []
    for i, cfg in enumerate(cfgs, 1):
        metrics_file = C.RUNS_DIR / cfg.run_id / "metrics.json"

        if metrics_file.exists() and not args.force:
            print(f"[{i}/{len(cfgs)}] {cfg.run_id} —— 已完成，跳过", flush=True)
            skipped.append(cfg.run_id)
            continue

        print(f"\n{'='*70}\n[{i}/{len(cfgs)}] {cfg.group} · {cfg.run_id} · {cfg.note}\n{'='*70}", flush=True)
        try:
            run_training(cfg)
            done.append(cfg.run_id)
        except Exception:
            print(f"[{cfg.run_id}] 失败：\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            failed.append(cfg.run_id)

    # 汇总
    print(f"\n{'='*70}\n执行结束：成功 {len(done)}，跳过 {len(skipped)}，失败 {len(failed)}")
    if failed:
        print(f"失败列表：{failed}")
    summary_file = C.RESULTS_DIR / "last_run_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"done": done, "skipped": skipped, "failed": failed}, f, ensure_ascii=False, indent=2)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
