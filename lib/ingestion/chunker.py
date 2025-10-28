"""
Layout-aware document chunking strategies.

Supports multiple chunking strategies:
- SEMANTIC: Chunk by headings/sections (keep structure intact)
- FIXED_SIZE: Fixed character count with overlap
- HYBRID: Semantic + fixed-size (split large sections)
- ATOMIC_ELEMENTS: Keep tables/figures as atomic chunks
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..retriever.base import Document
from .normalizer import NormalizedDocument


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    SEMANTIC = "semantic"  # chunk by headings/sections
    FIXED_SIZE = "fixed_size"  # fixed char count with overlap
    HYBRID = "hybrid"  # semantic + split large sections
    ATOMIC_ELEMENTS = "atomic_elements"  # tables/figures as atomic chunks


@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking.
    
    Attributes:
        strategy: Chunking strategy
        chunk_size: Target chunk size in characters (for fixed/hybrid)
        chunk_overlap: Overlap between chunks (for fixed/hybrid)
        min_chunk_size: Minimum chunk size (discard smaller)
        max_chunk_size: Maximum chunk size (split larger)
        keep_tables_atomic: Keep tables as single chunks
        keep_figures_atomic: Keep figures as single chunks
    """
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 3000
    keep_tables_atomic: bool = True
    keep_figures_atomic: bool = True


class DocumentChunker:
    """
    Chunk normalized documents using configurable strategies.
    
    Produces Document objects with content and metadata for indexing.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
    
    def chunk(self, normalized: NormalizedDocument) -> List[Document]:
        """
        Chunk a normalized document.
        
        Args:
            normalized: NormalizedDocument with Markdown content
            
        Returns:
            List of Document chunks with metadata
        """
        print(f"\n✂️  Chunking with strategy: {self.config.strategy.value}")
        
        if self.config.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(normalized)
        elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(normalized)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            chunks = self._chunk_hybrid(normalized)
        elif self.config.strategy == ChunkingStrategy.ATOMIC_ELEMENTS:
            chunks = self._chunk_atomic_elements(normalized)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
        
        # Filter by min size
        chunks = [c for c in chunks if len(c.content) >= self.config.min_chunk_size]
        
        print(f"  ✓ Created {len(chunks)} chunks")
        return chunks
    
    def _chunk_semantic(self, normalized: NormalizedDocument) -> List[Document]:
        """
        Chunk by semantic sections (headings).
        
        Each section (heading + content until next heading) becomes a chunk.
        """
        chunks = []
        current_section = []
        current_heading = None
        current_page = None
        
        for section in normalized.sections:
            section_type = section["type"]
            
            # Start new chunk at headings
            if section_type in ["title", "heading"]:
                # Save previous section
                if current_section:
                    chunk = self._create_chunk(
                        current_section,
                        heading=current_heading,
                        page=current_page,
                        filename=normalized.metadata.get("filename"),
                    )
                    chunks.append(chunk)
                
                # Start new section
                current_heading = section["content"]
                current_page = section["page"]
                current_section = [section]
            
            else:
                # Add to current section
                if not current_section:
                    current_heading = None
                    current_page = section["page"]
                
                current_section.append(section)
        
        # Save last section
        if current_section:
            chunk = self._create_chunk(
                current_section,
                heading=current_heading,
                page=current_page,
                filename=normalized.metadata.get("filename"),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_fixed_size(self, normalized: NormalizedDocument) -> List[Document]:
        """
        Chunk by fixed character count with overlap.
        
        Uses RecursiveCharacterTextSplitter on the full Markdown content.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        
        texts = text_splitter.split_text(normalized.content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Document(
                content=text,
                metadata={
                    "filename": normalized.metadata.get("filename"),
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    "strategy": "fixed_size",
                },
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_hybrid(self, normalized: NormalizedDocument) -> List[Document]:
        """
        Hybrid: semantic sections, but split large sections with fixed-size splitter.
        
        Preserves structure while ensuring no chunk exceeds max_chunk_size.
        """
        # First, chunk semantically
        semantic_chunks = self._chunk_semantic(normalized)
        
        # Then, split large chunks
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        
        for chunk in semantic_chunks:
            if len(chunk.content) > self.config.max_chunk_size:
                # Split large chunk
                sub_texts = text_splitter.split_text(chunk.content)
                for i, sub_text in enumerate(sub_texts):
                    sub_chunk = Document(
                        content=sub_text,
                        metadata={
                            **chunk.metadata,
                            "sub_chunk_index": i,
                            "total_sub_chunks": len(sub_texts),
                            "was_split": True,
                        },
                    )
                    chunks.append(sub_chunk)
            else:
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_atomic_elements(self, normalized: NormalizedDocument) -> List[Document]:
        """
        Keep tables and figures as atomic chunks; chunk text normally.
        
        - Tables: one chunk per table
        - Figures: one chunk per figure
        - Text: semantic or fixed-size chunking
        """
        chunks = []
        text_sections = []
        
        for section in normalized.sections:
            section_type = section["type"]
            
            # Atomic elements: table/figure
            if (section_type == "table" and self.config.keep_tables_atomic) or \
               (section_type == "figure" and self.config.keep_figures_atomic):
                # Save accumulated text
                if text_sections:
                    text_chunks = self._chunk_text_sections(text_sections, normalized.metadata)
                    chunks.extend(text_chunks)
                    text_sections = []
                
                # Add atomic element as single chunk
                chunk = Document(
                    content=section["content"],
                    metadata={
                        "filename": normalized.metadata.get("filename"),
                        "page": section["page"],
                        "type": section_type,
                        "heading": section.get("heading"),
                        "atomic": True,
                    },
                )
                chunks.append(chunk)
            
            else:
                # Accumulate text sections
                text_sections.append(section)
        
        # Process remaining text
        if text_sections:
            text_chunks = self._chunk_text_sections(text_sections, normalized.metadata)
            chunks.extend(text_chunks)
        
        return chunks
    
    def _chunk_text_sections(self, sections: List[Dict], doc_metadata: Dict) -> List[Document]:
        """Helper: chunk text sections using hybrid strategy."""
        # Reconstruct text from sections
        text = "\n".join(s["content"] for s in sections)
        
        # Split with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        
        # Create chunks
        chunks = []
        for i, chunk_text in enumerate(texts):
            chunk = Document(
                content=chunk_text,
                metadata={
                    "filename": doc_metadata.get("filename"),
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    "type": "text",
                },
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        sections: List[Dict],
        heading: Optional[str],
        page: int,
        filename: str,
    ) -> Document:
        """Create a Document chunk from sections."""
        # Combine section contents
        content_parts = []
        for section in sections:
            section_type = section["type"]
            content = section["content"]
            
            # Format based on type
            if section_type in ["title", "heading"]:
                content_parts.append(f"# {content}")
            elif section_type == "table":
                table_heading = section.get("heading", "Table")
                content_parts.append(f"**{table_heading}**\n\n{content}")
            elif section_type == "figure":
                fig_heading = section.get("heading", "Figure")
                content_parts.append(f"**{fig_heading}**: {content}")
            else:
                content_parts.append(content)
        
        chunk_content = "\n\n".join(content_parts)
        
        # Metadata
        metadata = {
            "filename": filename,
            "page": page,
            "heading": heading,
            "num_sections": len(sections),
            "strategy": "semantic",
        }
        
        return Document(
            content=chunk_content,
            metadata=metadata,
        )
