from app.ingest.chunk import MAX_CHARS, MIN_CHARS, TARGET_CHARS, chunk_pages

PARAGRAPH = "Net debt decreased over the year thanks to strong operating cash flow. " * 4  # ~280 chars


def _page(page: int, markdown: str) -> dict:
    return {"page": page, "markdown": markdown}


def test_headers_build_the_section_breadcrumb():
    pages = [_page(1, f"# Financial statements\n\n## Balance sheet\n\n{PARAGRAPH}\n\n### Net debt\n\n{PARAGRAPH}")]
    chunks = chunk_pages(pages)
    assert [c.section_path for c in chunks] == ["Financial statements > Balance sheet > Net debt"]


def test_new_top_level_section_starts_a_new_chunk_and_drops_deeper_levels():
    pages = [_page(1, f"# A\n\n### Deep\n\n{PARAGRAPH}\n\n# B\n\n{PARAGRAPH}")]
    chunks = chunk_pages(pages)
    assert [c.section_path for c in chunks] == ["A > Deep", "B"]


def test_chunks_stay_around_the_target_size_and_keep_page_ranges():
    pages = [_page(n, "\n\n".join([PARAGRAPH] * 4)) for n in range(1, 5)]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1
    assert all(MIN_CHARS <= len(c.content) <= TARGET_CHARS + MAX_CHARS for c in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 4
    assert all(c.page_start <= c.page_end for c in chunks)


def test_consecutive_chunks_overlap_by_the_last_block():
    blocks = [f"Block {i}. " + PARAGRAPH for i in range(12)]
    chunks = chunk_pages([_page(1, "\n\n".join(blocks))])
    assert len(chunks) >= 2
    first_tail = chunks[0].content.split("\n\n")[-1]
    assert chunks[1].content.startswith(first_tail)


def test_oversized_table_is_split_with_its_header_repeated():
    header = "| Item | 2025 | 2024 |\n|---|---|---|"
    rows = "\n".join(f"| Line item number {i} with a long label | {i * 1000} | {i * 900} |" for i in range(200))
    chunks = chunk_pages([_page(7, f"# Table\n\n{header}\n{rows}")])
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHARS + 200 for c in chunks)
    assert all(c.content.startswith("| Item | 2025 | 2024 |") for c in chunks)
    assert all((c.page_start, c.page_end) == (7, 7) for c in chunks)


def test_crumbs_are_merged_into_the_next_chunk_not_dropped():
    pages = [_page(1, "Short note."), _page(2, PARAGRAPH)]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].content.startswith("Short note.")
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
