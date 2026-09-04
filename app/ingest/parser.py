"""文档解析：PDF / DOCX / TXT / MD → 带结构的文本单元列表。

统一返回 [{text, page_no?, headers:[...]}]：
- text: 纯文本内容
- page_no: 仅 PDF 有效，其余为 None
- headers: 当前文本所属的标题链（从一级到当前级）
"""
from pathlib import Path


def parse_document(file_path: Path, file_type: str) -> list[dict]:
    ft = file_type.lower().lstrip(".")
    if ft == "pdf":
        return _parse_pdf(file_path)
    if ft == "docx":
        return _parse_docx(file_path)
    if ft in ("md", "markdown"):
        return _parse_markdown(file_path)
    if ft == "txt":
        return _parse_txt(file_path)
    raise ValueError(f"不支持的文件类型: {file_type}")


def _parse_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    units: list[dict] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            units.append({"text": text, "page_no": page_no, "headers": []})
    return units


def _parse_docx(path: Path) -> list[dict]:
    import docx

    document = docx.Document(str(path))
    units: list[dict] = []
    headers: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            units.append(
                {"text": "\n".join(buf).strip(), "page_no": None, "headers": list(headers)}
            )
            buf.clear()

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name if para.style else "") or ""
        level = _heading_level(style_name)
        if level:
            flush()
            headers = headers[: level - 1] + [text]
        else:
            buf.append(text)
    flush()
    return units


def _heading_level(style_name: str) -> int:
    """从 Word 样式名提取标题级别，如 'Heading 1' / '标题 1' -> 1。"""
    import re

    m = re.search(r"(heading|标题)\s*([1-6])", style_name, re.IGNORECASE)
    return int(m.group(2)) if m else 0


def _parse_markdown(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    units: list[dict] = []
    headers: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            units.append(
                {"text": "\n".join(buf).strip(), "page_no": None, "headers": list(headers)}
            )
            buf.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            headers = headers[: level - 1] + [title]
        else:
            buf.append(line)
    flush()
    return units


def _parse_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"text": text, "page_no": None, "headers": []}]
