"""Sentence-aware overlapping PDF text chunker preserving page and character metadata."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.pdf_processor import PageText, normalize_extracted_text


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    content: str
    char_start: int
    char_end: int


def chunk_pages(
    pages: list[PageText],
    *,
    chunk_size: int = 850,
    chunk_overlap: int = 160,
    min_chunk_length: int = 40,
) -> list[TextChunk]:
    size = max(200, chunk_size)
    overlap = min(size // 2, max(0, chunk_overlap))
    min_len = max(20, min_chunk_length)

    chunks: list[TextChunk] = []
    global_idx = 0

    for page in pages:
        cleaned = normalize_extracted_text(page.text)
        if not cleaned:
            continue

        if len(cleaned) <= size:
            chunks.append(
                TextChunk(
                    page_number=page.page_number,
                    chunk_index=global_idx,
                    content=cleaned,
                    char_start=0,
                    char_end=len(cleaned),
                )
            )
            global_idx += 1
            continue

        start = 0
        while start < len(cleaned):
            end = min(start + size, len(cleaned))

            if end < len(cleaned):
                window_start = max(start + int(size * 0.55), start)
                window = cleaned[window_start:end]

                para_break = window.rfind("\n\n")
                sent_break = max(
                    window.rfind(". "),
                    window.rfind("? "),
                    window.rfind("! "),
                    window.rfind(".\n"),
                )
                space_break = window.rfind(" ")

                if para_break > 0:
                    end = window_start + para_break + 2
                elif sent_break > 0:
                    end = window_start + sent_break + 2
                elif space_break > 0:
                    end = window_start + space_break + 1

            piece = cleaned[start:end].strip()
            if len(piece) >= min_len or not chunks:
                chunks.append(
                    TextChunk(
                        page_number=page.page_number,
                        chunk_index=global_idx,
                        content=piece,
                        char_start=start,
                        char_end=end,
                    )
                )
                global_idx += 1

            if end >= len(cleaned):
                break
            start = max(start + 1, end - overlap)

    return chunks
