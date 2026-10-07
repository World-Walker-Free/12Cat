"""模型库：ImageNet 预训练架构 + 可配置自定义 CNN。

为什么需要两种模型
------------------
1. 预训练架构（ResNet / VGG / MobileNet / EfficientNet）用于模型对比与
   "预训练 vs 从零"对照实验，直接调用 torchvision 并加载 ImageNet 权重。
2. 自定义 CNN 用于结构 / 激活函数消融。torchvision 预训练模型内部固定使用
   ReLU，无法干净地替换激活函数，因此激活函数与结构类消融必须在自定义
   网络上进行。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

# ImageNet 预训练权重来源（报告中需注明）：
# torchvision 官方权重，训练自 ImageNet-1K。
# 说明文档：https://pytorch.org/vision/stable/models.html
# 权重枚举：https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
WEIGHTS_SOURCE = "torchvision ImageNet-1K (IMAGENET1K_V1)"

# 支持的 torchvision 架构白名单
PRETRAINED_ARCHS = (
    "resnet18",
    "resnet34",
    "vgg11_bn",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "densenet121",
)

# 各架构分类头的属性路径，用于两阶段微调时冻结/解冻主干
_HEAD_PATHS: dict[str, tuple[str, ...]] = {
    "resnet18": ("fc",),
    "resnet34": ("fc",),
    "vgg11_bn": ("classifier", "6"),
    "mobilenet_v3_small": ("classifier", "3"),
    "efficientnet_b0": ("classifier", "1"),
    "densenet121": ("classifier",),
    "custom_cnn": ("head",),
}

# 激活函数注册表；inplace 仅对 ReLU / LeakyReLU 有效
ACTIVATIONS: dict[str, dict] = {
    "relu": {"cls": nn.ReLU, "kwargs": {"inplace": True}},
    "leaky_relu": {"cls": nn.LeakyReLU, "kwargs": {"negative_slope": 0.01, "inplace": True}},
    "sigmoid": {"cls": nn.Sigmoid, "kwargs": {}},
    "tanh": {"cls": nn.Tanh, "kwargs": {}},
}


def make_activation(name: str) -> nn.Module:
    """按名称构造激活函数模块。"""
    if name not in ACTIVATIONS:
        raise ValueError(f"未知激活函数 {name!r}，可选：{sorted(ACTIVATIONS)}")
    spec = ACTIVATIONS[name]
    return spec["cls"](**spec["kwargs"])


# ------------------------------------------------------------------ 自定义 CNN
class ConfigurableCNN(nn.Module):
    """结构可配置的卷积网络，用于结构与激活函数消融。

    参数
    ----
    num_conv_layers : 卷积层总数（控制深度）
    base_channels   : 第一层输出通道数，每经过一个池化阶段翻倍（控制宽度）
    pool_every      : 每多少层卷积后接一次最大池化
    activation      : 激活函数名称，见 ACTIVATIONS
    use_bn          : 是否使用 BatchNorm
    dropout_p       : 分类头前的 Dropout 概率

    结构：若干 [Conv3x3 -> (BN) -> 激活] + 周期性 MaxPool
          -> 全局平均池化 -> Dropout -> Linear
    """

    def __init__(
        self,
        num_classes: int = 12,
        num_conv_layers: int = 4,
        base_channels: int = 32,
        pool_every: int = 2,
        activation: str = "relu",
        use_bn: bool = True,
        dropout_p: float = 0.5,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = in_channels

        for i in range(num_conv_layers):
            out_ch = base_channels * (2 ** (i // pool_every))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(make_activation(activation))
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


# ------------------------------------------------------------------ 统一入口
def build_model(
    arch: str,
    num_classes: int = 12,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """按架构名构造模型。

    arch 为 PRETRAINED_ARCHS 中的名称时调用 torchvision；
    为 "custom_cnn" 时构造 ConfigurableCNN，额外参数由 kwargs 传入。
    """
    if arch == "custom_cnn":
        cnn_keys = {
            "num_conv_layers",
            "base_channels",
            "pool_every",
            "activation",
            "use_bn",
            "dropout_p",
        }
        cnn_kwargs = {k: v for k, v in kwargs.items() if k in cnn_keys}
        return ConfigurableCNN(num_classes=num_classes, **cnn_kwargs)

    if arch not in PRETRAINED_ARCHS:
        raise ValueError(f"未知架构 {arch!r}，可选：{PRETRAINED_ARCHS + ('custom_cnn',)}")

    builder = getattr(tvm, arch)
    if not pretrained:
        return builder(weights=None, num_classes=num_classes)

    # torchvision 在加载 ImageNet 权重时要求输出维度与权重一致（1000 类），
    # 因此先按原样加载，再把分类头替换为 num_classes 输出的新层。
    model = builder(weights="IMAGENET1K_V1")
    _replace_head(model, arch, num_classes)
    return model


def count_params(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_submodule(model: nn.Module, path: tuple[str, ...]) -> nn.Module:
    for attr in path:
        model = model[int(attr)] if attr.isdigit() else getattr(model, attr)
    return model


def _replace_head(model: nn.Module, arch: str, num_classes: int) -> None:
    """把分类头替换为新的 Linear(num_features, num_classes)。"""
    path = _HEAD_PATHS[arch]
    parent = _get_submodule(model, path[:-1])
    key = path[-1]
    old = parent[int(key)] if key.isdigit() else getattr(parent, key)
    new = nn.Linear(old.in_features, num_classes)
    if key.isdigit():
        parent[int(key)] = new
    else:
        setattr(parent, key, new)


def set_backbone_frozen(model: nn.Module, arch: str, frozen: bool) -> None:
    """冻结/解冻主干，仅保留分类头可训练（用于两阶段微调）。

    frozen=True 时冻结全部参数，再解冻分类头。
    """
    for p in model.parameters():
        p.requires_grad = not frozen
    if frozen:
        for p in _get_submodule(model, _HEAD_PATHS[arch]).parameters():
            p.requires_grad = True
