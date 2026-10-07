"""配置驱动的训练与验证脚本。

一次实验 = 一个 TrainConfig。所有实验共用同一套训练、评估、记录逻辑，
从而保证对比的公平性与结果的可比性。

每次实验在 results/runs/<run_id>/ 下产出：
    config.json          本次实验的完整配置
    history.csv          逐 epoch 的 loss / acc / 耗时 / 学习率
    metrics.json         汇总指标（准确率、参数量、耗时、收敛速度、过拟合）
    model.pt             验证集上最优的权重
    confusion_matrix.npy 最优模型在验证集上的混淆矩阵
    val_predictions.csv  最优模型的逐样本预测结果
    val_per_class.csv    最优模型的逐类 precision / recall / F1 / 样本数

注意：评估全程只使用从训练集划分出的验证集，不使用 cat_12_test/
中的 240 张无标签测试图。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config as C
from .data import build_dataloaders
from .models import WEIGHTS_SOURCE, build_model, count_params, set_backbone_frozen


# ------------------------------------------------------------------ 配置
@dataclass
class TrainConfig:
    """单次实验的完整配置。"""

    run_id: str
    arch: str
    pretrained: bool = False
    group: str = ""
    note: str = ""

    # 数据
    resolution: int = 224
    augment: bool = True

    # 优化
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"          # adam | sgd
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: str = "cosine"        # cosine | step | none
    label_smoothing: float = 0.0

    # 结构（仅 custom_cnn 生效）
    activation: str = "relu"
    use_bn: bool = True
    dropout_p: float = 0.5
    num_conv_layers: int = 4
    base_channels: int = 32
    pool_every: int = 2

    # 其它
    freeze_epochs: int = 0           # >0 时先冻结主干若干 epoch 再解冻
    num_workers: int = 4
    amp: bool = True
    seed: int = C.SEED

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 工具
def set_seed(seed: int) -> None:
    """固定全部随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"未知优化器 {cfg.optimizer!r}")


def build_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"未知学习率调度器 {cfg.scheduler!r}")


# ------------------------------------------------------------------ 训练/评估
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """训练一个 epoch，返回 (平均损失, 准确率)。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        n += targets.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """在验证集上评估，返回 (平均损失, 准确率, 预测, 真值)。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * targets.size(0)
        correct += (preds == targets).sum().item()
        n += targets.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    return total_loss / n, correct / n, all_preds, all_targets


# ------------------------------------------------------------------ 主流程
def run_training(cfg: TrainConfig) -> dict:
    """执行一次完整实验，写入结果目录并返回指标字典。"""
    C.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    set_seed(cfg.seed)

    run_dir = C.RUNS_DIR / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    # ---- 数据
    train_loader, val_loader, data_meta = build_dataloaders(
        resolution=cfg.resolution,
        batch_size=cfg.batch_size,
        augment=cfg.augment,
        num_workers=cfg.num_workers,
    )

    # ---- 模型
    model = build_model(
        arch=cfg.arch,
        num_classes=C.NUM_CLASSES,
        pretrained=cfg.pretrained,
        activation=cfg.activation,
        use_bn=cfg.use_bn,
        dropout_p=cfg.dropout_p,
        num_conv_layers=cfg.num_conv_layers,
        base_channels=cfg.base_channels,
        pool_every=cfg.pool_every,
    ).to(device)

    n_params = count_params(model)

    # 两阶段微调：先冻结主干， freeze_epochs 轮后解冻
    if cfg.freeze_epochs > 0 and cfg.pretrained:
        set_backbone_frozen(model, cfg.arch, frozen=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 训练循环
    history: list[dict] = []
    best_val_acc, best_epoch, best_state = -1.0, 0, None
    total_time = 0.0

    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_epochs > 0 and cfg.pretrained and epoch == cfg.freeze_epochs + 1:
            set_backbone_frozen(model, cfg.arch, frozen=False)

        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        epoch_time = time.perf_counter() - t0
        total_time += epoch_time

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "epoch_time_s": epoch_time,
            }
        )
        print(
            f"[{cfg.run_id}] epoch {epoch:3d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({epoch_time:.1f}s)",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- 用最优权重做最终评估
    model.load_state_dict(best_state)
    val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device, use_amp)
    cm = confusion_matrix(targets, preds, labels=list(range(C.NUM_CLASSES)))

    # 逐类验证集指标：prevalence 均衡，因此宏平均 F1 与准确率可直接对照
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(C.NUM_CLASSES)), zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    pd.DataFrame(
        {
            "class": C.CLASS_NAMES,
            "precision": precision.round(4),
            "recall": recall.round(4),
            "f1": f1.round(4),
            "support": support,
        }
    ).to_csv(run_dir / "val_per_class.csv", index=False)

    torch.save(best_state, run_dir / "model.pt")
    np.save(run_dir / "confusion_matrix.npy", cm)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({"target": targets, "pred": preds}).to_csv(
        run_dir / "val_predictions.csv", index=False
    )

    # ---- 收敛速度：首次达到最优验证准确率 95% 的 epoch
    threshold = 0.95 * best_val_acc
    epochs_to_converge = next(
        (h["epoch"] for h in history if h["val_acc"] >= threshold), best_epoch
    )

    best_row = history[best_epoch - 1]
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    )

    metrics = {
        "run_id": cfg.run_id,
        "group": cfg.group,
        "note": cfg.note,
        "arch": cfg.arch,
        "pretrained": cfg.pretrained,
        "weights_source": WEIGHTS_SOURCE if cfg.pretrained else "random init",
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "val_macro_f1": round(macro_f1, 4),
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "final_train_acc": history[-1]["train_acc"],
        "overfit_gap": best_row["train_acc"] - best_row["val_acc"],
        "epochs_to_converge": epochs_to_converge,
        "total_train_time_s": total_time,
        "mean_epoch_time_s": total_time / cfg.epochs,
        "peak_vram_mb": peak_vram_mb,
        "epochs": cfg.epochs,
        "resolution": cfg.resolution,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "optimizer": cfg.optimizer,
        "scheduler": cfg.scheduler,
        "activation": cfg.activation,
        "use_bn": cfg.use_bn,
        "dropout_p": cfg.dropout_p,
        "augment": cfg.augment,
        "num_conv_layers": cfg.num_conv_layers,
        "base_channels": cfg.base_channels,
        "freeze_epochs": cfg.freeze_epochs,
        "amp": use_amp,
        "seed": cfg.seed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        **{f"data_{k}": v for k, v in data_meta.items()},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"[{cfg.run_id}] DONE best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}, "
        f"params={n_params/1e6:.2f}M, time={total_time:.1f}s, "
        f"converge@{epochs_to_converge}",
        flush=True,
    )
    return metrics
