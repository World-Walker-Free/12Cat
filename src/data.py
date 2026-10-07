"""数据管线：读取标签、分层划分、数据集封装与数据增强。

设计要点
--------
1. 标签只从 data/train_list.txt 读取。cat_12_train/ 目录下有 2160 张图，
   但只有 2039 张有标签，直接按目录遍历会把 121 张无标签图当作数据。
2. 划分结果缓存到 results/split.json，所有实验复用同一份划分。
3. 验证集只做确定性的缩放与中心裁剪，不做任何随机增强。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config as C


# ------------------------------------------------------------------ 标签
def read_labels() -> list[tuple[str, int]]:
    """读取 train_list.txt，返回 [(相对路径, 类别)] 列表。

    文件每行格式为 ``cat_12_train/xxx.jpg<Tab>类别``。
    """
    items: list[tuple[str, int]] = []
    with open(C.LABEL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.split("\t")
            items.append((rel_path, int(label)))
    return items


# ------------------------------------------------------------------ 划分
def make_or_load_split(val_ratio: float = 0.2) -> dict[str, list[str]]:
    """分层划分训练/验证集，结果缓存复用。

    使用分层抽样（stratified）保证 12 个类别在训练集与验证集中的
    比例与原始分布一致。第一次调用时随机划分并写入 SPLIT_FILE，
    之后所有实验直接读取同一份划分，确保对比公平。

    返回 ``{"train": [...], "val": [...], "val_ratio": float, "seed": int}``
    """
    if C.SPLIT_FILE.exists():
        with open(C.SPLIT_FILE, encoding="utf-8") as f:
            return json.load(f)

    items = read_labels()
    paths = [p for p, _ in items]
    labels = [y for _, y in items]

    train_paths, val_paths = train_test_split(
        paths,
        test_size=val_ratio,
        stratify=labels,
        random_state=C.SEED,
        shuffle=True,
    )

    split = {
        "train": sorted(train_paths),
        "val": sorted(val_paths),
        "val_ratio": val_ratio,
        "seed": C.SEED,
    }
    C.ensure_dirs()
    with open(C.SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return split


# ------------------------------------------------------------------ 数据集
class CatDataset(Dataset):
    """猫十二分类数据集。

    只加载 split 中列出的样本，图像从 DATA_ROOT 下按相对路径解析。
    """

    def __init__(self, rel_paths: list[str], transform) -> None:
        self.rel_paths = rel_paths
        self.transform = transform
        label_map = dict(read_labels())
        self.labels = [label_map[p] for p in rel_paths]

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        path = C.DATA_ROOT / self.rel_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.labels[idx]


# ------------------------------------------------------------------ 变换
def build_transforms(resolution: int, augment: bool):
    """构造 (训练集变换, 验证集变换)。

    augment=False 时训练集也使用确定性变换，用于"数据增强"消融实验。
    """
    normalize = transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)

    val_tf = transforms.Compose(
        [
            transforms.Resize(int(resolution * 1.14)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_tf = val_tf

    return train_tf, val_tf


# ------------------------------------------------------------------ DataLoader
def build_dataloaders(
    resolution: int,
    batch_size: int,
    augment: bool = True,
    num_workers: int = 4,
):
    """按当前配置构造训练/验证 DataLoader。

    返回 ``(train_loader, val_loader, meta)``，meta 中带各类别样本数，
    供报告中的"划分方法与结果"使用。
    """
    split = make_or_load_split()
    train_tf, val_tf = build_transforms(resolution, augment)

    train_ds = CatDataset(split["train"], train_tf)
    val_ds = CatDataset(split["val"], val_tf)

    generator = torch.Generator()
    generator.manual_seed(C.SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    label_map = dict(read_labels())
    train_counts = np.bincount(
        [label_map[p] for p in split["train"]], minlength=C.NUM_CLASSES
    ).tolist()
    val_counts = np.bincount(
        [label_map[p] for p in split["val"]], minlength=C.NUM_CLASSES
    ).tolist()

    meta = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "resolution": resolution,
        "augment": augment,
    }
    return train_loader, val_loader, meta
