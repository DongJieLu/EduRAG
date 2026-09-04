"""生成 M2 验收用的示例中文 PDF。用法: python scripts/gen_sample_pdf.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

pdfmetrics.registerFont(TTFont("SimHei", r"C:/Windows/Fonts/simhei.ttf"))

SECTIONS = [
    ("Java 基础知识点", "Title"),
    ("面向对象三大特性", "Heading1"),
    ("封装是指将数据与操作数据的方法绑定在一起，隐藏对象内部实现细节，只暴露必要的接口。继承允许子类复用父类的字段与方法，并在此基础上扩展。多态则让同一操作作用于不同对象时产生不同行为。", "Normal"),
    ("集合框架", "Heading1"),
    ("Java 集合框架提供了常用的数据结构，包括 List、Set、Map 三大接口。List 是有序可重复的集合，Set 是无序不可重复的集合，Map 存储键值对。线程安全方面，可使用 ConcurrentHashMap 等并发容器。", "Normal"),
    ("并发编程", "Heading1"),
    ("Java 并发编程围绕线程、锁、以及并发工具类展开。synchronized 关键字提供互斥语义，volatile 保证可见性，JUC 包提供线程池、原子类、并发集合等丰富工具，帮助开发者写出高效且安全的并发程序。", "Normal"),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "docs" / "sample_java.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4)
    styles = getSampleStyleSheet()
    for name in ("Title", "Heading1", "Normal"):
        styles[name].fontName = "SimHei"
    story = []
    for text, style_name in SECTIONS:
        story.append(Paragraph(text, styles[style_name]))
        story.append(Spacer(1, 12))
    doc.build(story)
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
