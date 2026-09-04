"""分块器单元测试。"""
import pytest

import app.ingest.chunker as chunker
from app.ingest.chunker import chunk_document


def test_chunk_short_text_not_split():
    parsed = [{"text": "这是一段足够长的短文本", "headers": [], "page_no": None}]
    chunks = chunk_document(parsed, "recursive", 400, 80)
    assert len(chunks) == 1
    assert chunks[0].text == "这是一段足够长的短文本"


def test_chunk_long_text_splits_within_size():
    text = "这是一个句子。" * 100  # 700 字
    parsed = [{"text": text, "headers": [], "page_no": None}]
    chunks = chunk_document(parsed, "recursive", 100, 20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunk_metadata_inherits_title_and_page():
    text = "内容" * 60
    parsed = [{"text": text, "headers": ["第一章", "第一节"], "page_no": 3}]
    chunks = chunk_document(parsed, "recursive", 50, 10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["title"] == "第一节"
        assert c.metadata["page_no"] == 3


def test_chunk_index_is_sequential():
    text = "内容" * 60
    parsed = [{"text": text, "headers": [], "page_no": None}]
    chunks = chunk_document(parsed, "recursive", 50, 10)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_drops_text_shorter_than_min():
    parsed = [{"text": "abc", "headers": [], "page_no": None}]
    chunks = chunk_document(parsed, "recursive", 400, 80)
    assert chunks == []


def test_chunk_overlap_applied():
    text = "0123456789" * 30  # 300 字符，无分隔符
    parsed = [{"text": text, "headers": [], "page_no": None}]
    chunks = chunk_document(parsed, "recursive", 50, 10)
    assert len(chunks) > 1
    # 相邻 chunk 尾部与下一 chunk 头部重叠 overlap 个字符
    assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_chunk_exceeds_limit_raises(monkeypatch):
    monkeypatch.setattr(chunker, "_MAX_CHUNKS_PER_DOC", 3)
    text = "内容" * 100
    parsed = [{"text": text, "headers": [], "page_no": None}]
    with pytest.raises(ValueError):
        chunk_document(parsed, "recursive", 10, 2)


def test_chunk_cleans_whitespace_and_page_numbers():
    parsed = [
        {
            "text": "第一行内容\n\n\n12\n\n第二行 内容  有多个空格",
            "headers": [],
            "page_no": 1,
        }
    ]
    chunks = chunk_document(parsed, "recursive", 400, 80)
    assert len(chunks) == 1
    assert "12" not in chunks[0].text  # 孤立页码被清洗
    assert "   " not in chunks[0].text  # 多余空格被合并
