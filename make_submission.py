"""生成作业提交包。

只打包"报告 + 代码 + 图表"，排除虚拟环境、模型权重（1.1 GB）与数据集（208 MB）。

用法（在项目根目录执行）：
    python make_submission.py --name 张三 --class 计2401 --id 240112345

产出：
    提交包/cnn_homework_张三_计2401_240112345.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 需要打包的内容：(源路径, 打包后的相对路径)
INCLUDE = [
    # ---- 代码 ----
    ("cat12_full_pipeline.py", "cat12_full_pipeline.py"),      # 作业要求的单文件完整实现
    ("src", "src"),                                            # 25 次实验的完整框架
    ("md2docx.py", "md2docx.py"),                              # 报告 md -> docx 转换脚本
    ("requirements.txt", "requirements.txt"),
    ("README.md", "README.md"),
    # ---- 报告 ----
    ("实验报告.docx", "实验报告.docx"),
    ("实验报告.md", "实验报告.md"),
    # ---- 结果 ----
    ("results/figures", "results/figures"),
    ("results/tables", "results/tables"),
    ("results/summary.csv", "results/summary.csv"),
    ("results/split.json", "results/split.json"),
]

# 打包时额外排除的后缀 / 目录名
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".pt", ".pth", ".npy"}
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def collect() -> list[tuple[Path, Path]]:
    """收集待打包文件，返回 [(绝对路径, 包内相对路径)]。"""
    files: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in INCLUDE:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  [跳过] 不存在: {src_rel}")
            continue
        if src.is_file():
            files.append((src, Path(dst_rel)))
            continue
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if p.suffix in EXCLUDE_SUFFIX:
                continue
            files.append((p, Path(dst_rel) / p.relative_to(src)))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="生成作业提交包")
    ap.add_argument("--name", required=True, help="姓名")
    ap.add_argument("--class", dest="cls", required=True, help="班级")
    ap.add_argument("--id", dest="sid", required=True, help="学号")
    ap.add_argument("--out", default="提交包", help="输出目录")
    args = ap.parse_args()

    zip_name = f"cnn_homework_{args.name}_{args.cls}_{args.sid}.zip"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / zip_name

    files = collect()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, rel in files:
            zf.write(src, rel.as_posix())

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\n已生成: {zip_path.relative_to(ROOT)}")
    print(f"  {len(files)} 个文件, {size_mb:.1f} MB")
    print(f"\n邮件标题请使用: cnn_homework_{args.name}_{args.cls}_{args.sid}")


if __name__ == "__main__":
    main()
