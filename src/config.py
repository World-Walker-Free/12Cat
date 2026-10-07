"""全局配置：路径常量与所有实验必须共享的不变量。

集中定义这些内容的目的，是保证实验矩阵中的每一次训练都使用
完全相同的数据划分、随机种子与归一化参数，从而让不同模型 /
范式 / 超参数之间的对比是公平的。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "猫的十二分类"
TRAIN_DIR = DATA_ROOT / "cat_12_train"
TEST_DIR = DATA_ROOT / "cat_12_test"
LABEL_FILE = DATA_ROOT / "data" / "train_list.txt"

RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"

# 数据划分缓存：所有实验复用同一份划分，避免每次重新随机划分
SPLIT_FILE = RESULTS_DIR / "split.json"

# ---------------------------------------------------------------- 任务常量
NUM_CLASSES = 12
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# ---------------------------------------------------------------- 随机种子
SEED = 42

# ---------------------------------------------------------------- 归一化
# 与 torchvision 的 ImageNet 预训练权重保持一致；
# 从零训练的实验也沿用同一组参数，保证与预训练实验只差"初始化方式"。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    """创建结果目录（幂等）。"""
    for d in (RESULTS_DIR, RUNS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
