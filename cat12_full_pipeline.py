"""猫的十二分类 —— 完整单文件实现

本文件包含从数据预处理到模型训练、验证的全部步骤，可独立运行，
无需依赖项目中的其它模块：

    1. 数据读取与分层划分       read_labels() / make_split()
    2. 数据预处理与增强         build_transforms() / CatDataset
    3. 模型构建                 build_model()（含预训练权重的获取方式）
    4. 训练与验证               train_one_epoch() / evaluate() / run()
    5. 结果记录                 results/single_run/ 下输出指标、曲线数据与混淆矩阵

运行示例
--------
    # 预训练 ResNet18 微调（推荐配置）
    python cat12_full_pipeline.py --arch resnet18 --pretrained --lr 1e-4 --epochs 25

    # 同一架构从零训练（对照实验）
    python cat12_full_pipeline.py --arch resnet18 --lr 1e-3 --epochs 25

    # 自定义 CNN + 激活函数消融
    python cat12_full_pipeline.py --arch custom_cnn --activation leaky_relu

固定随机种子
------------
种子 42，固定 Python / NumPy / PyTorch / CUDA 全部随机源，
并关闭 cuDNN 自动算法选择，保证结果可复现。

预训练权重获取方式
------------------
通过 torchvision 加载官方 ImageNet-1K 预训练权重，权重枚举为
``IMAGENET1K_V1``；首次运行时由 torchvision 自动从
https://download.pytorch.org/models/ 下载并缓存至
``~/.cache/torch/hub/checkpoints/``。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================ 全局配置

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "猫的十二分类"
LABEL_FILE = DATA_ROOT / "data" / "train_list.txt"
OUTPUT_DIR = ROOT / "results" / "single_run"

NUM_CLASSES = 12
SEED = 42

# 与预训练权重匹配的 ImageNet 归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WEIGHTS_SOURCE = "torchvision ImageNet-1K 预训练权重（IMAGENET1K_V1）"

# 支持的 torchvision 架构
PRETRAINED_ARCHS = (
    "resnet18", "resnet34", "vgg11_bn",
    "mobilenet_v3_small", "efficientnet_b0",
)

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.01, inplace=True),
    "sigmoid": lambda: nn.Sigmoid(),
    "tanh": lambda: nn.Tanh(),
}


# ============================================================ 1. 随机种子
def set_seed(seed: int) -> None:
    """固定全部随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================ 2. 数据准备
def read_labels() -> list[tuple[str, int]]:
    """读取标签文件，返回 [(相对路径, 类别)]。

    每行格式：cat_12_train/xxx.jpg<Tab>类别(0~11)。
    注意：cat_12_train/ 下有 2160 张图，但仅 2039 张有标签，
    因此必须以标签文件为准，不能按目录遍历。
    """
    items: list[tuple[str, int]] = []
    with open(LABEL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split("\t")
            items.append((rel_path, int(label)))
    return items


def make_split(val_ratio: float = 0.2) -> tuple[list[str], list[str]]:
    """从训练集中按类别分层抽样划分训练集与验证集。

    分层抽样保证 12 个类别在训练集与验证集中的比例与原始分布一致，
    避免验证集类别分布波动导致的准确率估计不稳定。

    验证集仅从训练集划分，不使用 cat_12_test/。
    """
    items = read_labels()
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    train_paths, val_paths = train_test_split(
        paths, test_size=val_ratio, stratify=labels,
        random_state=SEED, shuffle=True,
    )
    return train_paths, val_paths


def build_transforms(resolution: int, augment: bool = True):
    """构造训练集与验证集的图像变换。

    训练集：随机裁剪 + 水平翻转 + 色彩抖动（数据增强）
    验证集：确定性的缩放与中心裁剪，不做任何随机增强
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    val_tf = transforms.Compose([
        transforms.Resize(int(resolution * 1.14)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize,
    ])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ]) if augment else val_tf

    return train_tf, val_tf


class CatDataset(Dataset):
    """猫的十二分类数据集。"""

    def __init__(self, rel_paths: list[str], label_map: dict[str, int], transform):
        self.rel_paths = rel_paths
        self.labels = [label_map[p] for p in rel_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        image = Image.open(DATA_ROOT / self.rel_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


# ============================================================ 3. 模型构建
class ConfigurableCNN(nn.Module):
    """可配置的卷积网络，用于结构与激活函数消融实验。

    结构：[Conv3x3 → (BatchNorm) → 激活] × num_conv_layers
          （每 pool_every 层后接一次 MaxPool）
          → 全局平均池化 → Dropout → 全连接
    """

    def __init__(self, num_classes=NUM_CLASSES, num_conv_layers=4,
                 base_channels=32, pool_every=2, activation="relu",
                 use_bn=True, dropout_p=0.5):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for i in range(num_conv_layers):
            out_ch = base_channels * (2 ** (i // pool_every))
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(ACTIVATIONS[activation]())
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout_p),
                                  nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(arch: str, pretrained: bool, **kwargs) -> nn.Module:
    """构建模型。

    预训练模型的权重获取方式：调用 torchvision 时传入
    ``weights="IMAGENET1K_V1"``，torchvision 会自动下载官方
    ImageNet-1K 权重；由于预训练权重的输出维度为 1000，
    这里先按原样加载，再替换分类头为 12 类输出。
    """
    if arch == "custom_cnn":
        return ConfigurableCNN(**kwargs)

    if arch not in PRETRAINED_ARCHS:
        raise ValueError(f"未知架构 {arch!r}")

    builder = getattr(tvm, arch)
    if not pretrained:
        return builder(weights=None, num_classes=NUM_CLASSES)

    model = builder(weights="IMAGENET1K_V1")   # 下载并加载 ImageNet 预训练权重
    if arch in ("resnet18", "resnet34"):
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif arch == "vgg11_bn":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    elif arch == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


# ============================================================ 4. 训练与验证
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
        correct += (logits.argmax(1) == targets).sum().item()
        n += targets.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """在给定数据集上评估，返回 (损失, 准确率, 预测, 真值)。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds_all: list[int] = []
    targets_all: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        preds = logits.argmax(1)
        total_loss += loss.item() * targets.size(0)
        correct += (preds == targets).sum().item()
        n += targets.size(0)
        preds_all.extend(preds.cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

    return total_loss / n, correct / n, preds_all, targets_all


# ============================================================ 5. 主流程
def run(args) -> dict:
    """执行一次完整的训练与验证。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    set_seed(SEED)

    # ---- 数据：分层划分 + 预处理 ----
    train_paths, val_paths = make_split(val_ratio=args.val_ratio)
    label_map = dict(read_labels())
    train_tf, val_tf = build_transforms(args.resolution, augment=not args.no_augment)

    train_loader = DataLoader(
        CatDataset(train_paths, label_map, train_tf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        CatDataset(val_paths, label_map, val_tf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"训练集 {len(train_paths)} 张 / 验证集 {len(val_paths)} 张"
          f"（分层抽样，验证集比例 {args.val_ratio}，种子 {SEED}）")
    print(f"设备 {device}，预训练权重来源：{WEIGHTS_SOURCE if args.pretrained else '未使用（从零训练）'}")

    # ---- 模型 ----
    model = build_model(
        args.arch, args.pretrained,
        activation=args.activation, use_bn=not args.no_bn,
        dropout_p=args.dropout, num_conv_layers=args.num_conv_layers,
        base_channels=args.base_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- 训练 ----
    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, 0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        epoch_time = time.perf_counter() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "epoch_time_s": epoch_time,
        })
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f}  ({epoch_time:.1f}s)", flush=True)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - start
    best_row = history[best_epoch - 1]

    # ---- 用验证集上最优的权重做最终评估 ----
    model.load_state_dict(best_state)
    val_loss, val_acc, preds, targets = evaluate(
        model, val_loader, criterion, device, use_amp)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    threshold = 0.95 * best_acc
    converge_epoch = next((h["epoch"] for h in history if h["val_acc"] >= threshold), best_epoch)

    metrics = {
        "arch": args.arch,
        "pretrained": args.pretrained,
        "weights_source": WEIGHTS_SOURCE if args.pretrained else "random init",
        "seed": SEED,
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "val_ratio": args.val_ratio,
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "val_macro_f1": float(np.mean(f1)),
        "overfit_gap": best_row["train_acc"] - best_row["val_acc"],
        "epochs_to_converge": converge_epoch,
        "total_train_time_s": total_time,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
    }

    # ---- 保存结果 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    np.save(OUTPUT_DIR / "confusion_matrix.npy", cm)

    print("\n" + "=" * 62)
    print(f"验证集准确率 : {best_acc * 100:.2f}%（第 {best_epoch} 轮）")
    print(f"验证宏平均 F1: {np.mean(f1) * 100:.2f}%")
    print(f"参数量       : {n_params / 1e6:.3f} M")
    print(f"过拟合 gap   : {metrics['overfit_gap'] * 100:.2f} 个百分点")
    print(f"收敛轮次     : {converge_epoch}（达到最优准确率 95%）")
    print(f"总训练耗时   : {total_time:.1f} s")
    print(f"结果已保存至 : {OUTPUT_DIR}")
    print("=" * 62)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="猫的十二分类 —— 完整训练与验证流程")
    p.add_argument("--arch", default="resnet18",
                   choices=(*PRETRAINED_ARCHS, "custom_cnn"))
    p.add_argument("--pretrained", action="store_true", help="加载 ImageNet 预训练权重")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true", help="关闭数据增强")
    # 以下仅对 custom_cnn 生效
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--no-bn", action="store_true")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num-conv-layers", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
