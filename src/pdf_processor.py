"""
pdf_processor.py
-----------------
Handles PDF text extraction and splits the extracted text into
overlapping chunks that are small enough to embed and retrieve
accurately.
"""

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
    """
    Extracts raw text from every page of a PDF.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A list where each element is the raw text of one page.
    """
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
    """
    Splits a block of text into overlapping word-based chunks.

    Overlap helps preserve context across chunk boundaries so an
    answer that straddles a split point isn't lost.

    Args:
        text: The text to split.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """
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
        start = end - overlap  # step forward, keeping overlap
    return chunks


def process_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Full pipeline: extract text page-by-page, then chunk each page,
    tagging every chunk with the page it came from.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Words per chunk (passed to chunk_text).
        overlap: Overlapping words between chunks.

    Returns:
        A list of Chunk objects ready to be embedded.
    """
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
