"""文档解析器单元测试（仅覆盖 txt/md，不依赖 pypdf/docx）。"""
from app.ingest.parser import parse_document


def test_parse_txt_single_unit(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello 世界", encoding="utf-8")
    units = parse_document(f, "txt")
    assert len(units) == 1
    assert units[0]["text"] == "hello 世界"
    assert units[0]["page_no"] is None
    assert units[0]["headers"] == []


def test_parse_markdown_builds_header_chain(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(
        "# 第一章\n\n第一章内容\n\n## 第一节\n\n第一节内容\n\n## 第二节\n\n第二节内容\n",
        encoding="utf-8",
    )
    units = parse_document(f, "md")
    # 第一个标题下无内容，应被跳过；实际得到两个有内容的单元
    assert len(units) == 3
    assert units[0]["headers"] == ["第一章"]
    assert units[1]["headers"] == ["第一章", "第一节"]
    assert units[2]["headers"] == ["第一章", "第二节"]


def test_parse_unsupported_type_raises(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_text("data", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError):
        parse_document(f, "xyz")
