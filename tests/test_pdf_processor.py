"""
Unit tests for src/pdf_processor.py. Run with:
    pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pdf_processor import chunk_text


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_respects_overlap():
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) >= 2
    # The end of the first chunk should overlap with the start of the second.
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert first_chunk_words[-1] != second_chunk_words[-1]
    assert second_chunk_words[0] in first_chunk_words


def test_chunk_text_covers_all_words():
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    all_chunked_words = set()
    for chunk in chunks:
        all_chunked_words.update(chunk.split())

    assert all_chunked_words == set(words)
