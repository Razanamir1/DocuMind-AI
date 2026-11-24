"""PyMuPDF (fitz) PDF text extraction, validation, filename sanitization, and valid PDF builder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.core.errors import DocumentError


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedPdf:
    page_count: int
    pages: list[PageText]
    total_characters: int


def normalize_extracted_text(raw: str) -> str:
    text = raw.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def sanitize_filename(raw_name: str) -> str:
    base = raw_name.replace("\\", "/").split("/")[-1] if raw_name else "document.pdf"
    cleaned = re.sub(r"[\x00-\x1f\x7f<>:\"/\\|?*]+", "_", base)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_{2,}", "_", cleaned).lstrip(".")[:140]
    if not cleaned:
        return "document.pdf"
    return cleaned if cleaned.lower().endswith(".pdf") else f"{cleaned}.pdf"


def is_pdf_magic_header(data: bytes) -> bool:
    if not data or len(data) < 5:
        return False
    return b"%PDF-" in data[:1024]


def resolve_safe_upload_path(upload_dir: Path, stored_filename: str) -> Path:
    base = upload_dir.resolve()
    safe_name = Path(stored_filename).name
    target = (base / safe_name).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise DocumentError("INVALID_PATH", "Invalid document file path.", 400) from exc
    return target


def extract_pdf_pages(pdf_bytes: bytes) -> ExtractedPdf:
    """Extracts text page-by-page using PyMuPDF (fitz). Rejects empty, corrupted, encrypted, or scanned PDFs."""
    if not pdf_bytes:
        raise DocumentError("EMPTY_PDF", "The uploaded PDF file is empty.", 400)

    if not is_pdf_magic_header(pdf_bytes):
        raise DocumentError(
            "INVALID_PDF",
            "File does not appear to be a valid PDF document.",
            422,
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentError(
            "CORRUPTED_PDF",
            "Unable to open this PDF file. It may be corrupted or malformed.",
            422,
        ) from exc

    try:
        if doc.needs_pass:
            raise DocumentError(
                "ENCRYPTED_PDF",
                "Password-protected PDFs cannot be indexed. Please unlock the PDF first.",
                422,
            )

        page_count = doc.page_count
        if page_count <= 0:
            raise DocumentError("EMPTY_PDF", "This PDF contains no pages.", 422)

        pages: list[PageText] = []
        total_chars = 0

        for idx in range(page_count):
            page = doc.load_page(idx)
            raw_text = page.get_text("text")
            cleaned = normalize_extracted_text(raw_text)
            total_chars += len(cleaned)
            pages.append(PageText(page_number=idx + 1, text=cleaned))

        if total_chars < 15:
            raise DocumentError(
                "SCANNED_PDF",
                "No extractable text was found in this PDF. It appears to be a scanned or image-only document, and OCR is not enabled.",
                422,
            )

        return ExtractedPdf(
            page_count=page_count,
            pages=pages,
            total_characters=total_chars,
        )
    finally:
        doc.close()


def create_sample_pdf_bytes(
    title: str, page_sections: list[dict[str, object]]
) -> bytes:
    """Creates a clean, real multi-page PDF using PyMuPDF for workspace seeding and tests."""
    doc = fitz.open()
    try:
        total_pages = len(page_sections)
        for index, section in enumerate(page_sections):
            page = doc.new_page(width=612, height=792)  # US Letter
            heading = str(section.get("heading", ""))
            lines = [str(item) for item in section.get("lines", [])]  # type: ignore[union-attr]

            # Subtle running header & divider rule
            if title:
                page.insert_text(
                    (56, 40),
                    title,
                    fontsize=9,
                    fontname="helv",
                    color=(0.42, 0.46, 0.54),
                )
                page.insert_text(
                    (506, 40),
                    f"Page {index + 1} of {total_pages}",
                    fontsize=9,
                    fontname="helv",
                    color=(0.42, 0.46, 0.54),
                )
                page.draw_line(
                    fitz.Point(56, 48),
                    fitz.Point(556, 48),
                    color=(0.85, 0.88, 0.94),
                    width=0.8,
                )

            y = 78.0
            if heading:
                page.insert_text(
                    (56, y),
                    heading,
                    fontsize=15.5,
                    fontname="hebo",
                    color=(0.08, 0.12, 0.22),
                )
                y += 28.0

            for paragraph in lines:
                if not paragraph.strip():
                    y += 8.0
                    continue
                is_subhead = paragraph.endswith(":") and len(paragraph) < 55
                font_name = "hebo" if is_subhead else "helv"
                font_size = 11.0 if is_subhead else 10.5
                color = (0.16, 0.22, 0.48) if is_subhead else (0.14, 0.17, 0.23)

                rect = fitz.Rect(56, y, 556, 745)
                rc = page.insert_textbox(
                    rect,
                    paragraph,
                    fontsize=font_size,
                    fontname=font_name,
                    color=color,
                    lineheight=1.42,
                )
                # Advance y based on line estimate
                est_lines = max(1, (len(paragraph) // 78) + 1)
                y += est_lines * (font_size * 1.48) + (6.0 if is_subhead else 10.0)
                if rc < 0 or y > 730:
                    break

        return doc.tobytes()
    finally:
        doc.close()
