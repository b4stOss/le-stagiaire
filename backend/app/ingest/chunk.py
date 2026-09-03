"""Structure-aware chunking of page-anchored OCR markdown.

Strategy (see README): split on markdown headers, accumulate blocks to ~450
tokens (~1800 chars), keep the section breadcrumb and page range on every
chunk, carry a small tail overlap between consecutive chunks so a fact
straddling a boundary is never lost.
"""

import re
from dataclasses import dataclass

TARGET_CHARS = 1800  # ~450 tokens
MAX_CHARS = 3600  # hard cap: oversized single blocks (big tables) get split
MIN_CHARS = 200  # don't emit crumbs
OVERLAP_MAX_CHARS = 400

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass
class Chunk:
    page_start: int
    page_end: int
    section_path: str
    content: str


@dataclass
class _Block:
    page: int
    text: str
    header_level: int | None = None  # set when the block is a markdown header


def _split_page_into_blocks(page_no: int, markdown: str) -> list[_Block]:
    """Split one page's markdown into blocks: headers, tables, paragraphs."""
    blocks: list[_Block] = []
    current: list[str] = []
    in_table = False

    def flush() -> None:
        nonlocal current, in_table
        text = "\n".join(current).strip()
        if text:
            blocks.append(_Block(page=page_no, text=text))
        current = []
        in_table = False

    for line in markdown.splitlines():
        header = HEADER_RE.match(line)
        if header:
            flush()
            blocks.append(_Block(page=page_no, text=line.strip(), header_level=len(header.group(1))))
            continue
        is_table_line = line.lstrip().startswith("|")
        if in_table != is_table_line:
            flush()
            in_table = is_table_line
        if line.strip():
            current.append(line)
        elif not in_table:
            flush()
    flush()
    return blocks


def _split_oversized(block: _Block) -> list[_Block]:
    """Split a block longer than MAX_CHARS on line boundaries (table rows)."""
    if len(block.text) <= MAX_CHARS:
        return [block]
    parts: list[_Block] = []
    lines = block.text.splitlines()
    header_lines = lines[:2] if len(lines) > 2 and lines[0].lstrip().startswith("|") else []
    buf: list[str] = []
    for line in lines:
        buf.append(line)
        if sum(len(row) + 1 for row in buf) >= MAX_CHARS:
            parts.append(_Block(page=block.page, text="\n".join(buf)))
            buf = list(header_lines)  # repeat table header for readability
    if buf and "\n".join(buf).strip() != "\n".join(header_lines).strip():
        parts.append(_Block(page=block.page, text="\n".join(buf)))
    return parts


def _enter_section(breadcrumb: dict[int, str], header: _Block) -> None:
    """Record a header in the breadcrumb and drop any deeper levels left from the previous section."""
    assert header.header_level is not None
    title = HEADER_RE.sub(r"\2", header.text).strip()
    breadcrumb[header.header_level] = title
    for deeper in [k for k in breadcrumb if k > header.header_level]:
        del breadcrumb[deeper]


def chunk_pages(pages: list[dict]) -> list[Chunk]:
    """pages: [{"page": 1, "markdown": "..."}] -> ordered chunks."""
    breadcrumb: dict[int, str] = {}
    chunks: list[Chunk] = []
    buf: list[_Block] = []
    buf_section = ""

    def section_path() -> str:
        return " > ".join(breadcrumb[k] for k in sorted(breadcrumb))

    def flush(carry_overlap: bool = True) -> None:
        nonlocal buf, buf_section
        content = "\n\n".join(b.text for b in buf).strip()
        if len(content) >= MIN_CHARS:
            chunks.append(
                Chunk(
                    page_start=min(b.page for b in buf),
                    page_end=max(b.page for b in buf),
                    section_path=buf_section,
                    content=content,
                )
            )
            tail = buf[-1]
            buf = [tail] if carry_overlap and len(tail.text) <= OVERLAP_MAX_CHARS else []
        elif content:
            # too small to stand alone: keep it, it will merge into the next chunk
            return
        else:
            buf = []

    for page in pages:
        for block in _split_page_into_blocks(page["page"], page["markdown"]):
            if block.header_level is not None:
                # new section: close the current chunk, update the breadcrumb
                if block.header_level <= 2:
                    flush(carry_overlap=False)
                _enter_section(breadcrumb, block)
                buf_section = section_path()
                continue
            for part in _split_oversized(block):
                if not buf:
                    buf_section = section_path()
                buf.append(part)
                if sum(len(b.text) for b in buf) >= TARGET_CHARS:
                    flush()
    flush(carry_overlap=False)
    return chunks
