from __future__ import annotations

import re
from typing import Iterable, List


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    parts = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            window = text[start:end]
            last_space = window.rfind(" ")
            if last_space > 0 and last_space > chunk_size // 2:
                end = start + last_space
        chunk_text = text[start:end].strip()
        if chunk_text:
            parts.append(chunk_text)
        if end >= text_length:
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return parts


def _chunk_paragraphs(content: str, base_metadata: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    chunks = []
    chunk_idx = 0
    for para in paragraphs:
        segments = [para] if len(para) <= chunk_size else _split_long_text(para, chunk_size, chunk_overlap)
        for seg in segments:
            chunk_idx += 1
            chunk_id = f"{base_metadata.get('doc_id', 'doc')}-chunk-{chunk_idx:04d}"
            chunks.append(
                {
                    "text": seg,
                    "metadata": {
                        **base_metadata,
                        "chunk_id": chunk_id,
                        "chunk_char_start": 0,
                        "chunk_char_end": len(seg),
                    },
                }
            )
    return chunks


def _chunk_qna(content: str, base_metadata: dict) -> list[dict]:
    pattern = re.compile(r"(?im)^question\s*\d+\s*:", re.MULTILINE)
    matches = list(pattern.finditer(content))
    if not matches:
        return _chunk_paragraphs(content, base_metadata, 10_000, 0)  # fallback to single chunk

    chunks = []
    positions = [m.start() for m in matches] + [len(content)]
    chunk_idx = 0
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        segment = content[start:end].strip()
        if not segment:
            continue
        chunk_idx += 1
        chunk_id = f"{base_metadata.get('doc_id', 'doc')}-qna-{chunk_idx:04d}"
        chunks.append(
            {
                "text": segment,
                "metadata": {
                    **base_metadata,
                    "chunk_id": chunk_id,
                    "chunk_char_start": start,
                    "chunk_char_end": end,
                },
            }
        )
    return chunks


def _chunk_csv_rows(csv_rows: list[dict], base_metadata: dict) -> list[dict]:
    chunks = []
    for row in csv_rows:
        row_idx = row.get("row_index", len(chunks))
        text = row.get("text", "").strip()
        if not text:
            continue
        chunk_id = f"{base_metadata.get('doc_id', 'doc')}-row-{row_idx:04d}"
        chunks.append(
            {
                "text": text,
                "metadata": {
                    **base_metadata,
                    "chunk_id": chunk_id,
                    "chunk_char_start": 0,
                    "chunk_char_end": len(text),
                    "csv_row": row_idx,
                },
            }
        )
    return chunks


def _looks_like_qna(content: str) -> bool:
    return bool(re.search(r"(?im)^question\s*\d+\s*:", content))


def chunk_corpus(
    documents: Iterable[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size // 2, 0)
    all_chunks: list[dict] = []
    for doc in documents:
        base_meta = doc["metadata"]
        doc_type = base_meta.get("doc_type", "").lower()
        if doc_type == "csv" and doc.get("csv_rows"):
            chunks = _chunk_csv_rows(doc["csv_rows"], base_meta)
        else:
            content = doc["content"]
            if _looks_like_qna(content):
                chunks = _chunk_qna(content, base_meta)
            else:
                chunks = _chunk_paragraphs(content, base_meta, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks
