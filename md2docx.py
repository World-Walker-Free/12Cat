"""把 实验报告.md 转换为 实验报告.docx。

流程：pandoc 转换 → python-docx 后处理（页面、字体、表格字号、图注样式）。

用法（在项目根目录执行）：
    .venv\\Scripts\\python.exe md2docx.py
"""

from pathlib import Path

import docx
import pypandoc
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

SRC = Path("实验报告.md")
DST = Path("实验报告.docx")

# 中文字体方案：正文宋体 / 标题黑体，西文用 Times New Roman
BODY_CJK = "宋体"
HEAD_CJK = "黑体"
BODY_LATIN = "Times New Roman"

# 样式 -> (西文字体, 中文字体, 字号, 是否加粗)
STYLE_FONTS = {
    "Normal": (BODY_LATIN, BODY_CJK, 10.5, None),
    "Title": (HEAD_CJK, HEAD_CJK, 20, True),
    "Heading 1": (HEAD_CJK, HEAD_CJK, 16, True),
    "Heading 2": (HEAD_CJK, HEAD_CJK, 14, True),
    "Heading 3": (HEAD_CJK, HEAD_CJK, 12, True),
    "Heading 4": (HEAD_CJK, HEAD_CJK, 11, True),
    "Image Caption": (BODY_LATIN, BODY_CJK, 9, None),
    "Table Caption": (BODY_LATIN, BODY_CJK, 9, None),
    "Caption": (BODY_LATIN, BODY_CJK, 9, None),
}

CAPTION_STYLES = ("Image Caption", "Table Caption", "Caption")


def set_style_font(style, latin: str, cjk: str, size: float, bold: bool | None) -> None:
    """设置样式的西文与中文字体（python-docx 不直接支持 eastAsia）。"""
    style.font.name = latin
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), cjk)


def table_font_size(n_cols: int) -> float:
    """按列数自适应字号：列越多字号越小，避免宽表在 A4 纵向下挤出页面。"""
    if n_cols <= 5:
        return 9.0
    if n_cols <= 8:
        return 8.0
    return 7.0


def style_tables(document) -> int:
    """表格内容居中、表头加粗、按列数自适应字号。"""
    for table in document.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        size = table_font_size(len(table.columns))
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_before = Pt(0)
                    para.paragraph_format.space_after = Pt(0)
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:
                        run.font.size = Pt(size)
                        run.font.name = BODY_LATIN
                        run.font.bold = i == 0
                        run._element.get_or_add_rPr().get_or_add_rFonts().set(
                            qn("w:eastAsia"), BODY_CJK
                        )
    return len(document.tables)


def style_captions(document) -> int:
    """图注 / 表注居中。"""
    n = 0
    for para in document.paragraphs:
        if para.style.name in CAPTION_STYLES:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            n += 1
    return n


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"未找到源文件：{SRC}")

    print(f"[1/3] pandoc 转换 {SRC} -> {DST} ...")
    pypandoc.convert_file(
        str(SRC),
        "docx",
        outputfile=str(DST),
        extra_args=["--standalone", "--toc", "--toc-depth=2"],
    )

    print("[2/3] 后处理：页面与字体 ...")
    document = docx.Document(str(DST))

    for section in document.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)

    styles = {s.name: s for s in document.styles}
    for name, (latin, cjk, size, bold) in STYLE_FONTS.items():
        if name in styles:
            set_style_font(styles[name], latin, cjk, size, bold)
        else:
            print(f"    (跳过不存在的样式: {name})")

    print("[3/3] 后处理：表格与图注 ...")
    n_tables = style_tables(document)
    n_captions = style_captions(document)

    document.save(str(DST))
    print(f"\n完成：{DST}  ({DST.stat().st_size / 1024:.0f} KB)")
    print(f"  表格 {n_tables} 个，图注 {n_captions} 条")


if __name__ == "__main__":
    main()
