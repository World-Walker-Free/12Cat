# 猫的十二分类 —— CNN 实验代码

使用卷积神经网络实现猫的 12 分类，并对网络结构、训练范式与超参数做对照实验。

> 本仓库**只包含代码**，不含实验报告、实验产物与数据集。

---

## 一、环境配置

Python 3.10+（本项目在 3.13.1 上验证）。

```powershell
# 创建虚拟环境
uv venv --python 3.13 .venv

# 安装 CUDA 版 PyTorch
# 注意：PyPI 上 Windows 版 torch 是 CPU-only，须使用专用源
uv pip install --python .venv\Scripts\python.exe `
    --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/ `
    torch==2.9.1+cu128 torchvision

# 安装其余依赖
uv pip install --python .venv\Scripts\python.exe `
    numpy pandas matplotlib pillow scikit-learn tqdm
```

验证环境：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

无 GPU 时使用 CPU 版 torch 同样可以运行，但速度较慢，建议调小 `experiments.py` 中的 `EPOCHS`。

**预训练权重**由 torchvision 在首次运行时自动下载（ImageNet-1K，权重枚举 `IMAGENET1K_V1`），缓存于 `~/.cache/torch/hub/checkpoints/`，无需手动准备。

---

## 二、代码文件说明

| 文件 | 作用 |
| --- | --- |
| `cat12_full_pipeline.py` | **单文件完整实现**，含数据预处理、模型构建、训练与验证的全部步骤，可独立运行 |
| `src/config.py` | 全局常量：路径、类别数、随机种子、ImageNet 归一化参数 |
| `src/data.py` | 数据管线：标签读取、分层划分（结果缓存复用）、Dataset、数据增强 |
| `src/models.py` | 模型库：5 个预训练架构的加载与分类头替换、可配置自定义 CNN |
| `src/train.py` | 训练核心：`TrainConfig` 配置类、固定随机种子、训练/验证循环、指标记录 |
| `src/experiments.py` | 25 次实验的矩阵定义，每组只改变一个因素 |
| `src/run_matrix.py` | 批量执行器，支持按组/按实验筛选与断点续跑 |
| `src/summarize.py` | 汇总已有结果，生成对比表、曲线图与混淆矩阵 |

**设计要点**：一次实验对应一个 `TrainConfig`，所有实验共用同一套训练代码路径，因此"相同数据划分与训练设置"由代码结构保证，而非人工维护。全部实验复用 `results/split.json` 中缓存的同一份分层划分（种子 42）。

---

## 三、运行方式

### 数据准备

数据需自行放置于项目根目录的 `猫的十二分类/` 下（本仓库未包含）：

```
猫的十二分类/
├── cat_12_train/          2160 张图（其中 2039 张有标签）
├── cat_12_test/           240 张图（无标签，本实验不使用）
└── data/train_list.txt    2039 行，格式：cat_12_train/xxx.jpg<Tab>类别(0~11)
```

> 标签只以 `train_list.txt` 为准。`cat_12_train/` 下有 2160 张图但仅 2039 张有标签，
> 按目录遍历会把无标签图误当数据。

### 单文件运行

```powershell
# 预训练 ResNet18 微调（推荐配置）
.venv\Scripts\python.exe cat12_full_pipeline.py --arch resnet18 --pretrained --lr 1e-4 --epochs 25

# 同架构从零训练（对照实验）
.venv\Scripts\python.exe cat12_full_pipeline.py --arch resnet18 --lr 1e-3 --epochs 25
```

### 完整实验矩阵

所有命令在**项目根目录**执行：

```powershell
# 查看实验清单
.venv\Scripts\python.exe -m src.run_matrix --list

# 先跑单个实验验证流程（约 5 分钟）
.venv\Scripts\python.exe -m src.run_matrix --only a1_resnet18

# 按组运行
.venv\Scripts\python.exe -m src.run_matrix --groups A1
.venv\Scripts\python.exe -m src.run_matrix --groups A1 A2 B C1 C2 C3 C4 C5

# 运行完整实验矩阵（25 次，约 1.5 ~ 2.5 小时）
.venv\Scripts\python.exe -m src.run_matrix --all

# 汇总结果、生成表格与图表
.venv\Scripts\python.exe -m src.summarize
```

### 说明

- **断点续跑**：已完成的实验（存在 `metrics.json`）会自动跳过，中断后重跑同一命令即可继续；加 `--force` 可强制重跑。
- **依赖顺序**：C1/C2/C5 组的基准是 `a1_resnet18`（A1 组），C3/C4 组的基准是 `a2_customcnn`（A2 组），因此 **A1、A2 需先于 C 组运行**。使用 `--all` 时顺序已自动满足。
- **输出位置**：每次实验的配置、指标、权重与混淆矩阵写入 `results/runs/<实验名>/`；汇总表与图表写入 `results/tables/` 与 `results/figures/`。
- **快速试跑**：加 `--epochs 2` 可用 2 个 epoch 快速验证流程，但试跑结果会被记录，之后正式运行需加 `--force` 重跑或删除 `results/runs/` 目录。

---

## 四、实验设计

| 组 | 实验数 | 内容 |
| --- | --- | --- |
| A1 | 5 | 预训练模型对比（ResNet18/34、VGG11-BN、MobileNetV3-S、EfficientNet-B0） |
| A2 | 4 | 同架构从零训练，超参与 A1 完全一致 |
| B | 1 | 从零训练 + 调整学习率 |
| C1 | 2 | 学习率消融 |
| C2 | 3 | 批量大小消融 |
| C3 | 3 | 激活函数消融 |
| C4 | 5 | 网络结构消融（深度 / 宽度 / BatchNorm / Dropout） |
| C5 | 2 | 输入分辨率与数据增强消融 |

**公平性保证**：所有实验复用同一份分层划分（`results/split.json`，种子 42）；
A1 与 A2 除初始化方式外超参完全一致；C 组每组只有一个因素与基准不同。
