import io
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from app.config import OCR_CACHE_DIR, settings
from app.mistral import get_client

MAX_UPLOAD_BYTES = 45 * 1024 * 1024  # API limit is ~50 MB per document; stay under it


def _inline_tables(page) -> str:
    """Tables come back as separate attachments referenced as [tbl-N.md](tbl-N.md)
    in the page markdown; put their content back at the reference point."""
    markdown = page.markdown
    for table in page.tables or []:
        ref = f"[{table.id}]({table.id})"
        if ref in markdown:
            markdown = markdown.replace(ref, f"\n{table.content}\n")
        else:  # reference missing: keep the data anyway
            markdown += f"\n\n{table.content}\n"
    return markdown


def _ocr_one(pdf_name: str, pdf_bytes: bytes) -> list[dict]:
    client = get_client()
    uploaded = client.files.upload(
        file={"file_name": pdf_name, "content": pdf_bytes},
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id)
    result = client.ocr.process(
        model=settings.ocr_model,
        document={"type": "document_url", "document_url": signed.url},
        table_format="markdown",
    )
    return [{"page": p.index + 1, "markdown": _inline_tables(p)} for p in result.pages]


def _split_pdf(pdf_path: Path, parts: int) -> list[bytes]:
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    per_part = -(-total // parts)  # ceil
    out: list[bytes] = []
    for start in range(0, total, per_part):
        writer = PdfWriter()
        for page in reader.pages[start : start + per_part]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        out.append(buf.getvalue())
    return out


def run_ocr(pdf_path: Path) -> list[dict]:
    """OCR a PDF with mistral-ocr-latest. Returns [{"page": 1, "markdown": "..."}].

    Results are cached to data/ocr/<name>.json: OCR is the expensive step
    (~$4/1000 pages), it must never run twice for the same file.
    Files above the API's ~50 MB upload limit are split into page-range parts
    and stitched back together with correct page numbers.
    """
    cache_file = OCR_CACHE_DIR / f"{pdf_path.stem}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    size = pdf_path.stat().st_size
    if size <= MAX_UPLOAD_BYTES:
        pages = _ocr_one(pdf_path.name, pdf_path.read_bytes())
    else:
        parts = -(-size // MAX_UPLOAD_BYTES)
        pages = []
        for i, part_bytes in enumerate(_split_pdf(pdf_path, parts)):
            offset = len(pages)
            part_pages = _ocr_one(f"{pdf_path.stem}-part{i + 1}.pdf", part_bytes)
            pages.extend({"page": offset + p["page"], "markdown": p["markdown"]} for p in part_pages)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(pages, ensure_ascii=False))
    return pages
