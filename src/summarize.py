"""结果汇总与可视化。

读取 results/runs/*/metrics.json，产出报告所需的全部材料：

    results/summary.csv          全部实验的指标总表
    results/tables/*.md          分组对比表（Markdown，可直接粘贴进报告）
    results/figures/*.png        各类曲线图与混淆矩阵

图表统一使用英文标签，避免中文字体缺失导致乱码；报告正文用中文即可。
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

# 消融组对应的基准实验（该组中"未改变因素"的那一次）
BASELINES = {
    "C1": "a1_resnet18",
    "C2": "a1_resnet18",
    "C3": "a2_customcnn",
    "C4": "a2_customcnn",
    "C5": "a1_resnet18",
}

# 基准实验在曲线图中的图例名（run_id 本身看不出它是基准）
BASELINE_LABELS = {
    "C1": "lr=1e-3 (baseline)",
    "C2": "batch=64 (baseline)",
    "C3": "ReLU (baseline)",
    "C4": "baseline structure",
    "C5": "224px + aug (baseline)",
}

GROUP_TITLES = {
    "A1": "A1 · Pretrained model comparison",
    "A2": "A2 · From-scratch model comparison",
    "B": "B · From-scratch with tuned learning rate",
    "C1": "C1 · Learning rate",
    "C2": "C2 · Batch size",
    "C3": "C3 · Activation function",
    "C4": "C4 · Network structure",
    "C5": "C5 · Input resolution / augmentation",
}

# 报告表格中展示的列
TABLE_COLS = [
    ("run_id", "实验"),
    ("arch", "网络结构"),
    ("pretrained", "预训练"),
    ("best_val_acc", "验证准确率"),
    ("val_macro_f1", "验证宏F1"),
    ("best_epoch", "最优轮次"),
    ("epochs_to_converge", "收敛轮次"),
    ("params_m", "参数量(M)"),
    ("mean_epoch_time_s", "每轮耗时(s)"),
    ("total_train_time_s", "总耗时(s)"),
    ("overfit_gap", "过拟合gap"),
    ("note", "说明"),
]


# ------------------------------------------------------------------ 读取
def load_runs() -> pd.DataFrame:
    """读取全部已完成实验的 metrics.json。"""
    rows = []
    for metrics_file in sorted(C.RUNS_DIR.glob("*/metrics.json")):
        with open(metrics_file, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(
            f"未找到任何实验结果。请先运行实验：\n"
            f"  python -m src.run_matrix --all\n"
            f"（结果目录：{C.RUNS_DIR}）"
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["group", "run_id"]).reset_index(drop=True)


def load_history(run_id: str) -> pd.DataFrame:
    """读取某次实验的逐 epoch 训练历史。"""
    path = C.RUNS_DIR / run_id / "history.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def steady_epoch_time(run_id: str) -> float:
    """稳态每轮耗时：第 2 轮起的中位数。

    首轮包含 cuDNN 算子预热与文件系统缓存冷启动，耗时可达稳态的 10~40 倍，
    直接用均值会显著高估。因此图表与报告统一采用剔除首轮后的中位数。
    """
    hist = load_history(run_id)
    if len(hist) < 2:
        return float("nan")
    return float(hist["epoch_time_s"].iloc[1:].median())


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    """把指标表格式化为适合写入报告的字符串表。"""
    # 只保留实际存在的列，兼容早期未记录宏 F1 的实验结果
    cols = [(c, t) for c, t in TABLE_COLS if c in df.columns]
    out = df[[c for c, _ in cols]].copy()
    rename = dict(cols)
    out["pretrained"] = out["pretrained"].map({True: "是", False: "否"})
    out["best_val_acc"] = (out["best_val_acc"] * 100).round(2)
    if "val_macro_f1" in out.columns:
        out["val_macro_f1"] = (out["val_macro_f1"].astype(float) * 100).round(2)
    out["overfit_gap"] = (out["overfit_gap"] * 100).round(2)
    out["mean_epoch_time_s"] = out["mean_epoch_time_s"].round(2)
    out["total_train_time_s"] = out["total_train_time_s"].round(1)
    return out.rename(columns=rename)


def to_markdown(df: pd.DataFrame) -> str:
    """手写 Markdown 表格，避免依赖 tabulate。"""
    fmt = _fmt(df)
    header = "| " + " | ".join(fmt.columns) + " |"
    sep = "| " + " | ".join("---" for _ in fmt.columns) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in fmt.itertuples(index=False)
    ]
    return "\n".join([header, sep, *body])


# ------------------------------------------------------------------ 绘图工具
def _style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--")
    # 仅在有带标签的图元时绘制图例（柱状图等无需图例）
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)


def _save(fig, name: str) -> None:
    C.ensure_dirs()
    path = C.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  图: {path.relative_to(C.ROOT)}")


# ------------------------------------------------------------------ 各类图
def plot_curves(
    run_ids: list[str], out_name: str, title: str, labels: dict[str, str] | None = None
) -> None:
    """绘制指定实验的验证准确率与损失曲线。

    labels 可选，用于把 run_id 映射为更易读的图例名（如基准实验）。
    """
    labels = labels or {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for run_id in run_ids:
        hist = load_history(run_id)
        if hist.empty:
            continue
        name = labels.get(run_id, run_id)
        axes[0].plot(hist["epoch"], hist["val_acc"] * 100, marker="o", ms=3, label=name)
        axes[1].plot(hist["epoch"], hist["val_loss"], marker="o", ms=3, label=name)
        axes[2].plot(hist["epoch"], hist["train_acc"] * 100, marker="o", ms=3, label=name)
    _style_axes(axes[0], "Validation accuracy", "epoch", "accuracy (%)")
    _style_axes(axes[1], "Validation loss", "epoch", "loss")
    _style_axes(axes[2], "Training accuracy", "epoch", "accuracy (%)")
    fig.suptitle(title)
    _save(fig, out_name)


def plot_group_bars(df: pd.DataFrame, group: str) -> None:
    """消融组：准确率 + 过拟合 gap + 每轮耗时 的柱状对比。"""
    sub = df[df["group"] == group].copy()
    if sub.empty:
        return
    base_id = BASELINES.get(group)
    if base_id and (df["run_id"] == base_id).any():
        sub = pd.concat([df[df["run_id"] == base_id], sub]).drop_duplicates("run_id")

    sub = sub.sort_values("best_val_acc", ascending=False)
    sub = sub.assign(epoch_time=[steady_epoch_time(r) for r in sub["run_id"]])
    labels = sub["run_id"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].bar(x, sub["best_val_acc"] * 100, color="#4C72B0")
    axes[0].set_ylim(0, 100)
    _style_axes(axes[0], "Best validation accuracy", "", "accuracy (%)")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")

    axes[1].bar(x, sub["overfit_gap"] * 100, color="#C44E52")
    _style_axes(axes[1], "Overfit gap (train - val acc)", "", "gap (pp)")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")

    axes[2].bar(x, sub["epoch_time"], color="#55A868")
    _style_axes(axes[2], "Steady epoch time", "", "seconds")
    axes[2].set_xticks(x, labels, rotation=30, ha="right")

    fig.suptitle(GROUP_TITLES.get(group, group))
    _save(fig, f"group_{group.lower()}.png")


def plot_model_comparison(df: pd.DataFrame) -> None:
    """A1 模型对比：准确率 / 参数量 / 每轮耗时。"""
    sub = df[df["group"] == "A1"].sort_values("best_val_acc", ascending=False)
    if sub.empty:
        return
    sub = sub.assign(epoch_time=[steady_epoch_time(r) for r in sub["run_id"]])
    labels = sub["run_id"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].bar(x, sub["best_val_acc"] * 100, color="#4C72B0")
    axes[0].set_ylim(0, 100)
    _style_axes(axes[0], "Best validation accuracy", "", "accuracy (%)")

    axes[1].bar(x, sub["params_m"], color="#8172B3")
    _style_axes(axes[1], "Parameter count", "", "million params")

    axes[2].bar(x, sub["epoch_time"], color="#55A868")
    _style_axes(axes[2], "Steady epoch time", "", "seconds")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
    fig.suptitle("A1 · Pretrained model comparison")
    _save(fig, "a1_model_comparison.png")


def plot_pretrained_vs_scratch(df: pd.DataFrame) -> None:
    """预训练 vs 从零：按架构配对的对照图。"""
    pairs = []
    for arch in df[df["group"] == "A1"]["arch"].unique():
        a1 = df[(df["group"] == "A1") & (df["arch"] == arch)]
        a2 = df[(df["group"] == "A2") & (df["arch"] == arch)]
        if not a1.empty and not a2.empty:
            pairs.append((arch, a1.iloc[0]["run_id"], a2.iloc[0]["run_id"]))
    if not pairs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    labels = [p[0] for p in pairs]
    x = np.arange(len(pairs))
    w = 0.38
    pre = [df[df["run_id"] == p[1]]["best_val_acc"].iloc[0] * 100 for p in pairs]
    scr = [df[df["run_id"] == p[2]]["best_val_acc"].iloc[0] * 100 for p in pairs]

    axes[0].bar(x - w / 2, pre, w, label="pretrained", color="#4C72B0")
    axes[0].bar(x + w / 2, scr, w, label="from scratch", color="#C44E52")
    axes[0].set_ylim(0, 100)
    _style_axes(axes[0], "Best validation accuracy", "", "accuracy (%)")

    pre_c = [df[df["run_id"] == p[1]]["epochs_to_converge"].iloc[0] for p in pairs]
    scr_c = [df[df["run_id"] == p[2]]["epochs_to_converge"].iloc[0] for p in pairs]
    axes[1].bar(x - w / 2, pre_c, w, label="pretrained", color="#4C72B0")
    axes[1].bar(x + w / 2, scr_c, w, label="from scratch", color="#C44E52")
    _style_axes(axes[1], "Epochs to converge (95% of best)", "", "epochs")

    pre_g = [df[df["run_id"] == p[1]]["overfit_gap"].iloc[0] * 100 for p in pairs]
    scr_g = [df[df["run_id"] == p[2]]["overfit_gap"].iloc[0] * 100 for p in pairs]
    axes[2].bar(x - w / 2, pre_g, w, label="pretrained", color="#4C72B0")
    axes[2].bar(x + w / 2, scr_g, w, label="from scratch", color="#C44E52")
    _style_axes(axes[2], "Overfit gap at best epoch", "", "gap (pp)")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
    fig.suptitle("Pretrained vs from-scratch (identical split and hyperparameters)")
    _save(fig, "pretrained_vs_scratch.png")


def plot_pretrained_vs_scratch_curves(df: pd.DataFrame) -> None:
    """预训练 vs 从零的验证准确率曲线叠加对比（按架构配对）。

    与 plot_pretrained_vs_scratch 的柱状图互补：柱状图给出最终数值，
    本图给出整个训练过程中验证准确率的上升过程，直接展示收敛速度差异。
    """
    pairs = []
    for arch in df[df["group"] == "A1"]["arch"].unique():
        a1 = df[(df["group"] == "A1") & (df["arch"] == arch)]
        a2 = df[(df["group"] == "A2") & (df["arch"] == arch)]
        if not a1.empty and not a2.empty:
            pairs.append((arch, a1.iloc[0]["run_id"], a2.iloc[0]["run_id"]))
    if not pairs:
        return

    axes = np.atleast_1d(
        plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.2), squeeze=False)[1][0]
    )
    for ax, (arch, pre_id, scr_id) in zip(axes, pairs):
        for run_id, ls, label in (
            (pre_id, "-", "pretrained"),
            (scr_id, "--", "from scratch"),
        ):
            hist = load_history(run_id)
            if hist.empty:
                continue
            ax.plot(hist["epoch"], hist["val_acc"] * 100, ls, marker="o", ms=3, label=label)
        ax.set_title(arch, fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val accuracy (%)")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)

    fig = axes[0].figure
    fig.suptitle("Validation accuracy curve: pretrained vs from-scratch")
    _save(fig, "pretrained_vs_scratch_curves.png")


def plot_paired_pretrained_curves(df: pd.DataFrame) -> None:
    """A2 组各架构对应的预训练模型训练曲线。

    与 curves_a2.png（A2 组的从零曲线）并列使用，使实验二的预训练侧
    与从零侧各有一张同口径的曲线图。仅包含在 A1 中有预训练对照的架构
    （自定义 CNN 无预训练对应项，故不出现）。
    """
    ids = [
        a1.iloc[0]["run_id"]
        for arch in df[df["group"] == "A2"]["arch"].unique()
        if not (a1 := df[(df["group"] == "A1") & (df["arch"] == arch)]).empty
    ]
    if not ids:
        return
    plot_curves(ids, "curves_a1_paired.png", "Pretrained models paired with A2 group")


def plot_confusion_matrix(df: pd.DataFrame, run_id: str | None = None) -> None:
    """最优模型的混淆矩阵（也可指定某个实验）。"""
    if run_id is None:
        run_id = df.loc[df["best_val_acc"].idxmax(), "run_id"]
    cm_file = C.RUNS_DIR / run_id / "confusion_matrix.npy"
    if not cm_file.exists():
        return

    cm = np.load(cm_file)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, data, title, fmt in (
        (axes[0], cm, f"Confusion matrix (counts) · {run_id}", "d"),
        (axes[1], cm_norm, f"Confusion matrix (row-normalized) · {run_id}", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(C.NUM_CLASSES))
        ax.set_yticks(range(C.NUM_CLASSES))
        ax.set_xticklabels(C.CLASS_NAMES, rotation=90, fontsize=7)
        ax.set_yticklabels(C.CLASS_NAMES, fontsize=7)
        thresh = data.max() / 2 if data.max() > 0 else 0
        for i in range(C.NUM_CLASSES):
            for j in range(C.NUM_CLASSES):
                ax.text(
                    j, i, format(data[i, j], fmt),
                    ha="center", va="center", fontsize=6,
                    color="white" if data[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "confusion_matrix_best.png")


# ------------------------------------------------------------------ 主流程
def write_tables(df: pd.DataFrame) -> None:
    """写出总表与分组表（Markdown）。"""
    tables_dir = C.RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(C.RESULTS_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"  表: {(C.RESULTS_DIR / 'summary.csv').relative_to(C.ROOT)}")

    lines = ["# 实验总表\n", to_markdown(df), ""]
    for group in sorted(df["group"].unique()):
        sub = df[df["group"] == group].sort_values("best_val_acc", ascending=False)
        lines += [f"\n## {GROUP_TITLES.get(group, group)}\n", to_markdown(sub), ""]

    path = tables_dir / "all_tables.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  表: {path.relative_to(C.ROOT)}")


def write_split_summary() -> None:
    """写出数据划分说明（报告需要报告划分方法与结果）。"""
    if not C.SPLIT_FILE.exists():
        return
    with open(C.SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    from .data import read_labels

    label_map = dict(read_labels())
    tr = np.bincount([label_map[p] for p in split["train"]], minlength=C.NUM_CLASSES)
    va = np.bincount([label_map[p] for p in split["val"]], minlength=C.NUM_CLASSES)
    n_labeled = len(split["train"]) + len(split["val"])
    n_on_disk = len(list(C.TRAIN_DIR.glob("*.jpg")))
    n_test = len(list(C.TEST_DIR.glob("*.jpg")))

    lines = [
        "# 验证集划分方法与数据说明\n",
        "## 1. 可用数据",
        "",
        f"- `cat_12_train/` 共 {n_on_disk} 张图像，其中 **{n_labeled} 张有标签**（标签来自 `data/train_list.txt`）",
        f"- 已排除的无标签图像：{n_on_disk - n_labeled} 张（不在标签文件中，不参与训练与评估）",
        "",
        "## 2. 划分方法",
        "",
        f"- 划分对象：**仅从 {n_labeled} 张有标签训练图中划分**验证集，验证集不来自测试集",
        "- 划分方式：**按类别分层抽样**（stratified split），保证 12 个类别在训练集与验证集中的"
        "比例与原始分布一致",
        f"- 实现：`sklearn.model_selection.train_test_split(test_size={split['val_ratio']}, "
        f"stratify=标签, random_state={split['seed']})`",
        f"- 划分比例：验证集占 {split['val_ratio']:.0%}，即训练集 **{len(split['train'])}** 张 / "
        f"验证集 **{len(split['val'])}** 张",
        f"- 随机种子：{split['seed']}",
        "- 可复现性：划分结果固化在 `results/split.json`，全部实验复用**同一份划分**，"
        "使不同模型与超参数之间的对比不受划分差异干扰",
        "",
        "## 3. 划分结果（每类样本数）",
        "",
        "| 类别 | 训练集 | 验证集 | 合计 |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(C.NUM_CLASSES):
        lines.append(f"| class_{i} | {tr[i]} | {va[i]} | {tr[i] + va[i]} |")
    lines += [
        f"| **合计** | **{tr.sum()}** | **{va.sum()}** | **{tr.sum() + va.sum()}** |",
        "",
        "各类别在训练集与验证集中的数量比例一致，说明分层抽样生效。",
        "",
        "## 4. 关于测试集",
        "",
        f"`cat_12_test/` 的 {n_test} 张图像**没有标签**，本实验全程未使用、未参与任何评测；"
        "报告中所有指标均为**验证集指标**。相关代码仅保留路径常量定义，"
        "数据管线与评估流程均未加载该目录。",
        "",
    ]
    tables_dir = C.RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "data_split.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  表: {path.relative_to(C.ROOT)}")


def write_validation_report(df: pd.DataFrame) -> None:
    """写出验证集指标报告。

    仅汇总验证集指标，不包含任何测试集内容。
    """
    tables_dir = C.RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    best_id = df.loc[df["best_val_acc"].idxmax(), "run_id"]
    best = df[df["run_id"] == best_id].iloc[0]

    lines = [
        "# 验证集指标报告\n",
        "> 以下全部指标均在**从训练集划分出的验证集**上计算，"
        "未使用 `cat_12_test/` 中的无标签测试图。\n",
        "## 1. 最优模型\n",
        f"- 实验：`{best_id}`（{best['arch']}，"
        f"{'ImageNet 预训练 + 微调' if best['pretrained'] else '从零训练'}）",
        f"- 验证准确率：**{best['best_val_acc'] * 100:.2f}%**（第 {int(best['best_epoch'])} 轮）",
    ]
    if "val_macro_f1" in df.columns:
        lines.append(f"- 验证宏平均 F1：**{float(best['val_macro_f1']) * 100:.2f}%**")
    lines += [
        f"- 参数量：{best['params_m']:.3f} M",
        f"- 收敛轮次：{int(best['epochs_to_converge'])}（首次达到最优验证准确率 95% 所需的 epoch）",
        f"- 过拟合 gap：{best['overfit_gap'] * 100:.2f} 个百分点"
        "（最优轮次处 训练准确率 − 验证准确率）",
        "",
        "## 2. 各实验验证集指标汇总\n",
        to_markdown(df.sort_values("best_val_acc", ascending=False)),
        "",
    ]

    per_class_file = C.RUNS_DIR / best_id / "val_per_class.csv"
    if per_class_file.exists():
        per_class = pd.read_csv(per_class_file)
        lines += [
            f"## 3. 最优模型逐类验证集指标（{best_id}）\n",
            "| " + " | ".join(per_class.columns) + " |",
            "| " + " | ".join("---" for _ in per_class.columns) + " |",
        ]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in per_class.itertuples(index=False)]
        lines += [
            "",
            f"- 混淆矩阵：`results/figures/confusion_matrix_best.png`",
            f"- 逐样本预测：`results/runs/{best_id}/val_predictions.csv`",
            "",
        ]

    path = tables_dir / "validation_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  表: {path.relative_to(C.ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总实验结果为表格与图表")
    parser.add_argument("--confusion-run", default=None, help="指定用于混淆矩阵的实验（默认取最优）")
    args = parser.parse_args(argv)

    C.ensure_dirs()
    df = load_runs()
    print(f"读取到 {len(df)} 次实验，分组：{sorted(df['group'].unique())}")

    print("\n[1/4] 写出总表、划分说明与验证集指标报告 ...")
    write_tables(df)
    write_split_summary()
    write_validation_report(df)

    print("\n[2/4] 模型对比图 ...")
    plot_model_comparison(df)
    plot_pretrained_vs_scratch(df)
    plot_pretrained_vs_scratch_curves(df)
    plot_paired_pretrained_curves(df)

    print("\n[3/4] 各组曲线与消融图 ...")
    for group in sorted(df["group"].unique()):
        ids = df[df["group"] == group]["run_id"].tolist()
        labels: dict[str, str] = {}
        base_id = BASELINES.get(group)
        # 消融组的曲线同样需要画出基准，否则缺少对照
        if base_id and base_id not in ids and (df["run_id"] == base_id).any():
            ids = [base_id, *ids]
            labels[base_id] = BASELINE_LABELS[group]
        plot_curves(ids, f"curves_{group.lower()}.png", GROUP_TITLES.get(group, group), labels)
        plot_group_bars(df, group)

    print("\n[4/4] 最优模型混淆矩阵 ...")
    best = args.confusion_run or df.loc[df["best_val_acc"].idxmax(), "run_id"]
    plot_confusion_matrix(df, best)

    print(f"\n完成。最优实验：{best}（验证准确率 {df['best_val_acc'].max()*100:.2f}%）")
    print(f"全部产出位于：{C.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
