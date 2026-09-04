"""文档分块：recursive / markdown 策略，输出 Chunk（text + metadata）。

分块算法等价于递归字符切分：按分隔符优先级（段落→换行→句读→空格）在窗口内
寻找切点，overlap 通过窗口滑动实现。chunk_size/overlap 以字符计（中文语境近似 token）。
"""
from dataclasses import dataclass, field

# 分隔符优先级：从粗到细，尽量在语义边界切分
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

_MIN_CHUNK_LEN = 10        # 入库前丢弃长度小于该值的 chunk
_MAX_CHUNKS_PER_DOC = 2000  # 单文档 chunk 数上限


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_document(
    parsed: list[dict],
    strategy: str = "recursive",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """把 parsed 单元切分为 chunk。

    - recursive：忽略标题结构，按分隔符递归切分，metadata 继承来源单元。
    - markdown：title 取 headers 末级标题链，超大块再递归切分。
    """
    chunks: list[Chunk] = []
    for unit in parsed:
        text = (unit.get("text") or "").strip()
        if not text:
            continue
        title = _last_header(unit.get("headers") or [])
        page_no = unit.get("page_no")
        for piece in _split_text(text, chunk_size, chunk_overlap):
            piece = _clean(piece)
            if len(piece) < _MIN_CHUNK_LEN:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    metadata={
                        "title": title,
                        "page_no": page_no,
                        "chunk_index": len(chunks),
                    },
                )
            )
            if len(chunks) > _MAX_CHUNKS_PER_DOC:
                raise ValueError(
                    f"单个文档 chunk 数超过上限 {_MAX_CHUNKS_PER_DOC}，请拆分文档"
                )
    return chunks


def _last_header(headers: list[str]) -> str:
    return headers[-1] if headers else ""


def _clean(text: str) -> str:
    """入库前清洗：去多余空白、去孤立页码/页眉脚（启发式）。"""
    import re

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去纯数字页码行（如单独一行的 "12"）
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    return text.strip()


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = _find_cut(text, start + chunk_size // 2, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _find_cut(text: str, lo: int, hi: int) -> int:
    """在 [lo, hi) 内找最后一个分隔符位置，返回切点（分隔符结束位置）。"""
    best = -1
    for sep in _SEPARATORS:
        idx = text.rfind(sep, lo, hi)
        if idx > best:
            best = idx + len(sep)
    return best
