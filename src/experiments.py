"""实验矩阵定义。

共 25 次实验，分 8 组，覆盖作业要求的全部对比维度：

    A1  预训练模型对比（5 种架构，同设置）
    A2  从零训练对比（4 种架构，超参与 A1 完全一致）
    B   预训练 vs 从零的公平性补充（从零 + 调整后的学习率）
    C1  学习率消融
    C2  批量大小消融
    C3  激活函数消融（在自定义 CNN 上，因预训练模型内部固定 ReLU）
    C4  网络结构消融（深度 / 宽度 / BatchNorm / Dropout）
    C5  输入分辨率与数据增强消融

设计原则：同一组内只有一个因素不同，其余参数从该组基准继承，
确保"每次只改变一个因素"。
"""

from __future__ import annotations

from .train import TrainConfig

# 统一训练轮数。A 组内预训练与从零使用相同轮数，保证对比公平。
# 取 25 是为了让从零训练也有充分收敛机会，避免"从零表现差只是因为轮数不够"的质疑。
EPOCHS = 25
RESOLUTION = 224
BATCH_SIZE = 64
LR = 1e-3
NUM_WORKERS = 4

# A 组基准：预训练 ResNet18，后续消融均以此为基准只改一个因素
A_GROUP_ARCHS = ["resnet18", "resnet34", "vgg11_bn", "mobilenet_v3_small", "efficientnet_b0"]
SCRATCH_ARCHS = ["resnet18", "resnet34", "mobilenet_v3_small", "custom_cnn"]

# 自定义 CNN 的默认结构（A2 中作为从零训练的轻量基准）
CUSTOM_CNN_DEFAULTS = {
    "num_conv_layers": 4,
    "base_channels": 32,
    "pool_every": 2,
    "activation": "relu",
    "use_bn": True,
    "dropout_p": 0.5,
}

_SHORT = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "vgg11_bn": "vgg11bn",
    "mobilenet_v3_small": "mobilenetv3s",
    "efficientnet_b0": "efficientnetb0",
    "custom_cnn": "customcnn",
}


def _base(**overrides) -> dict:
    """所有实验的公共基准参数。"""
    cfg = {
        "resolution": RESOLUTION,
        "augment": True,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "optimizer": "adam",
        "scheduler": "cosine",
        "num_workers": NUM_WORKERS,
        "amp": True,
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def all_experiments() -> list[TrainConfig]:
    """构造完整的 25 次实验配置。"""
    cfgs: list[TrainConfig] = []

    # ---------------- A1：预训练模型对比 ----------------
    for arch in A_GROUP_ARCHS:
        cfgs.append(
            TrainConfig(
                run_id=f"a1_{_SHORT[arch]}",
                arch=arch,
                pretrained=True,
                group="A1",
                note="ImageNet 预训练 + 微调，同数据划分与超参",
                **_base(),
            )
        )

    # ---------------- A2：从零训练对比（超参与 A1 完全一致）----------------
    for arch in SCRATCH_ARCHS:
        extra = CUSTOM_CNN_DEFAULTS if arch == "custom_cnn" else {}
        cfgs.append(
            TrainConfig(
                run_id=f"a2_{_SHORT[arch]}",
                arch=arch,
                pretrained=False,
                group="A2",
                note="从零训练，超参与 A1 相同，仅初始化方式不同",
                **_base(**extra),
            )
        )

    # ---------------- B：公平性补充（从零 + 更大学习率）----------------
    cfgs.append(
        TrainConfig(
            run_id="b_scratch_resnet18_lr3e3",
            arch="resnet18",
            pretrained=False,
            group="B",
            note="从零训练但使用更大学习率，检验超参调优能否弥补无预训练的劣势",
            **_base(lr=3e-3),
        )
    )

    # ---------------- C1：学习率（基准 1e-3 见 a1_resnet18）----------------
    for lr in (1e-2, 1e-4):
        cfgs.append(
            TrainConfig(
                run_id=f"c1_lr{lr:g}",
                arch="resnet18",
                pretrained=True,
                group="C1",
                note=f"学习率={lr:g}",
                **_base(lr=lr),
            )
        )

    # ---------------- C2：批量大小（基准 64 见 a1_resnet18）----------------
    for bs in (16, 32, 128):
        cfgs.append(
            TrainConfig(
                run_id=f"c2_bs{bs}",
                arch="resnet18",
                pretrained=True,
                group="C2",
                note=f"批量大小={bs}",
                **_base(batch_size=bs),
            )
        )

    # ---------------- C3：激活函数（基准 relu 见 a2_customcnn）----------------
    for act in ("leaky_relu", "sigmoid", "tanh"):
        cfgs.append(
            TrainConfig(
                run_id=f"c3_{act}",
                arch="custom_cnn",
                pretrained=False,
                group="C3",
                note=f"激活函数={act}",
                **_base(**{**CUSTOM_CNN_DEFAULTS, "activation": act}),
            )
        )

    # ---------------- C4：网络结构（基准见 a2_customcnn）----------------
    cfgs.append(
        TrainConfig(
            run_id="c4_depth6",
            arch="custom_cnn",
            pretrained=False,
            group="C4",
            note="加深：卷积层数 4 -> 6",
            **_base(**{**CUSTOM_CNN_DEFAULTS, "num_conv_layers": 6}),
        )
    )
    cfgs.append(
        TrainConfig(
            run_id="c4_depth2",
            arch="custom_cnn",
            pretrained=False,
            group="C4",
            note="减浅：卷积层数 4 -> 2",
            **_base(**{**CUSTOM_CNN_DEFAULTS, "num_conv_layers": 2}),
        )
    )
    cfgs.append(
        TrainConfig(
            run_id="c4_ch64",
            arch="custom_cnn",
            pretrained=False,
            group="C4",
            note="加宽：基础通道数 32 -> 64",
            **_base(**{**CUSTOM_CNN_DEFAULTS, "base_channels": 64}),
        )
    )
    cfgs.append(
        TrainConfig(
            run_id="c4_nobn",
            arch="custom_cnn",
            pretrained=False,
            group="C4",
            note="关闭 BatchNorm",
            **_base(**{**CUSTOM_CNN_DEFAULTS, "use_bn": False}),
        )
    )
    cfgs.append(
        TrainConfig(
            run_id="c4_nodropout",
            arch="custom_cnn",
            pretrained=False,
            group="C4",
            note="关闭 Dropout",
            **_base(**{**CUSTOM_CNN_DEFAULTS, "dropout_p": 0.0}),
        )
    )

    # ---------------- C5：输入分辨率与数据增强 ----------------
    cfgs.append(
        TrainConfig(
            run_id="c5_res160",
            arch="resnet18",
            pretrained=True,
            group="C5",
            note="输入分辨率 224 -> 160",
            **_base(resolution=160),
        )
    )
    cfgs.append(
        TrainConfig(
            run_id="c5_noaug",
            arch="resnet18",
            pretrained=True,
            group="C5",
            note="关闭数据增强",
            **_base(augment=False),
        )
    )

    return cfgs


def select(groups: list[str] | None = None) -> list[TrainConfig]:
    """按组名筛选实验；groups 为 None 时返回全部。"""
    cfgs = all_experiments()
    if groups:
        wanted = {g.upper() for g in groups}
        cfgs = [c for c in cfgs if c.group.upper() in wanted]
    return cfgs


if __name__ == "__main__":
    for c in all_experiments():
        print(f"{c.group:4s} {c.run_id:28s} arch={c.arch:20s} pretrained={c.pretrained}")
