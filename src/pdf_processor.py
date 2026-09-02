from dataclasses import dataclass
from typing import List
import pdfplumber


@dataclass
class Chunk:
    """A single chunk of text extracted from a PDF, with metadata."""
    text: str
    page_number: int
    chunk_id: int


def extract_text_by_page(pdf_path: str) -> List[str]:
    pages_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return pages_text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  
    return chunks


def process_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Chunk]:
    pages = extract_text_by_page(pdf_path)
    all_chunks: List[Chunk] = []
    chunk_id = 0

    for page_number, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text, chunk_size, overlap):
            all_chunks.append(
                Chunk(text=piece, page_number=page_number, chunk_id=chunk_id)
            )
            chunk_id += 1

    return all_chunks
