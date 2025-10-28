"""
Multimodal PDF ingestion pipeline for research documents.

This module provides tools to extract, normalize, chunk, and index
PDF documents with tables, figures, and text using Azure Document Intelligence
and GPT-4o for figure captioning.

Components:
- extractor: Extract text, tables, figures from PDFs via Azure Document Intelligence
- normalizer: Convert extracted content to normalized Markdown
- chunker: Layout-aware chunking strategies (by heading, fixed-size, hybrid)
"""

from .extractor import PDFExtractor
from .normalizer import MarkdownNormalizer
from .chunker import DocumentChunker, ChunkingStrategy

__all__ = [
    "PDFExtractor",
    "MarkdownNormalizer",
    "DocumentChunker",
    "ChunkingStrategy",
]
