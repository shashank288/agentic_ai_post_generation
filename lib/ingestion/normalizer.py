"""
Normalize extracted PDF content to structured Markdown.

Converts text blocks, tables, and figures into a unified Markdown format
with preserved structure (headings, paragraphs, tables, captions).
"""

from typing import List, Dict
from dataclasses import dataclass

from .extractor import ExtractedContent


@dataclass
class NormalizedDocument:
    """
    Normalized document with Markdown content and metadata.
    
    Attributes:
        content: Full Markdown text
        sections: List of section dicts with {heading, content, page, type}
        metadata: Document metadata
    """
    content: str
    sections: List[Dict]
    metadata: Dict


class MarkdownNormalizer:
    """
    Convert extracted PDF content to normalized Markdown.
    
    Structure:
    - Headings: # Title, ## Section
    - Paragraphs: plain text
    - Tables: Markdown table format
    - Figures: **Figure N**: <caption>
    """
    
    def __init__(self, include_page_markers: bool = True):
        """
        Initialize normalizer.
        
        Args:
            include_page_markers: Add `<!-- Page N -->` markers
        """
        self.include_page_markers = include_page_markers
    
    def normalize(self, extracted: ExtractedContent) -> NormalizedDocument:
        """
        Convert extracted content to Markdown.
        
        Args:
            extracted: ExtractedContent from PDFExtractor
            
        Returns:
            NormalizedDocument with structured Markdown
        """
        print("\n📝 Normalizing to Markdown...")
        
        # Merge all content into a unified timeline sorted by page/position
        timeline = self._build_timeline(extracted)
        
        # Convert timeline to Markdown sections
        sections = []
        markdown_lines = []
        current_page = 0
        
        for item in timeline:
            # Page marker
            if self.include_page_markers and item["page"] != current_page:
                current_page = item["page"]
                markdown_lines.append(f"\n<!-- Page {current_page} -->\n")
            
            # Convert item to markdown
            md_text, section_data = self._item_to_markdown(item)
            markdown_lines.append(md_text)
            
            if section_data:
                sections.append(section_data)
        
        content = "\n".join(markdown_lines)
        
        print(f"  ✓ Normalized {len(sections)} sections ({len(content)} chars)")
        
        return NormalizedDocument(
            content=content,
            sections=sections,
            metadata=extracted.metadata,
        )
    
    def _build_timeline(self, extracted: ExtractedContent) -> List[Dict]:
        """
        Merge text blocks, tables, figures into a single timeline sorted by page/position.
        
        Returns:
            List of items with {type, content, page, ...}
        """
        timeline = []
        
        # Add text blocks
        for block in extracted.text_blocks:
            timeline.append({
                "type": block["type"],  # title, sectionHeading, paragraph
                "content": block["content"],
                "page": block["page"],
                "sort_key": (block["page"], self._get_position(block.get("bounding_regions"))),
            })
        
        # Add tables
        for table in extracted.tables:
            timeline.append({
                "type": "table",
                "content": table["content"],
                "page": table["page"],
                "metadata": {
                    "row_count": table["row_count"],
                    "column_count": table["column_count"],
                    "index": table["index"],
                },
                "sort_key": (table["page"], self._get_position(table.get("bounding_regions"))),
            })
        
        # Add figures
        for fig in extracted.figures:
            timeline.append({
                "type": "figure",
                "content": fig["caption"],
                "page": fig["page"],
                "metadata": {
                    "index": fig["index"],
                },
                "sort_key": (fig["page"], self._get_position_from_bbox(fig.get("bbox"))),
            })
        
        # Sort by page, then vertical position
        timeline.sort(key=lambda x: x["sort_key"])
        
        return timeline
    
    def _get_position(self, bounding_regions) -> float:
        """Get vertical position (y0) from bounding regions."""
        if not bounding_regions or len(bounding_regions) == 0:
            return 0.0
        
        region = bounding_regions[0]
        polygon = region.polygon if hasattr(region, "polygon") else []
        
        if len(polygon) >= 2:
            # polygon is [x0, y0, x1, y1, x2, y2, x3, y3]
            return polygon[1]  # y0
        
        return 0.0
    
    def _get_position_from_bbox(self, bbox) -> float:
        """Get vertical position from bbox (x0, y0, x1, y1)."""
        if bbox and len(bbox) >= 2:
            return bbox[1]  # y0
        return 0.0
    
    def _item_to_markdown(self, item: Dict) -> tuple:
        """
        Convert a timeline item to Markdown text and section metadata.
        
        Returns:
            (markdown_text, section_dict or None)
        """
        item_type = item["type"]
        content = item["content"]
        page = item["page"]
        
        # Headings
        if item_type == "title":
            md_text = f"# {content}\n"
            section = {
                "heading": content,
                "content": content,
                "page": page,
                "type": "title",
            }
            return md_text, section
        
        elif item_type == "sectionHeading":
            md_text = f"## {content}\n"
            section = {
                "heading": content,
                "content": content,
                "page": page,
                "type": "heading",
            }
            return md_text, section
        
        # Paragraphs
        elif item_type == "paragraph":
            md_text = f"{content}\n"
            section = {
                "heading": None,
                "content": content,
                "page": page,
                "type": "paragraph",
            }
            return md_text, section
        
        # Tables
        elif item_type == "table":
            table_idx = item["metadata"].get("index", 0)
            md_text = f"\n**Table {table_idx + 1}**\n\n{content}\n"
            section = {
                "heading": f"Table {table_idx + 1}",
                "content": content,
                "page": page,
                "type": "table",
                "metadata": item["metadata"],
            }
            return md_text, section
        
        # Figures
        elif item_type == "figure":
            fig_idx = item["metadata"].get("index", 0)
            caption = content
            md_text = f"\n**Figure {fig_idx + 1}**: {caption}\n"
            section = {
                "heading": f"Figure {fig_idx + 1}",
                "content": caption,
                "page": page,
                "type": "figure",
                "metadata": item["metadata"],
            }
            return md_text, section
        
        # Unknown type
        else:
            md_text = f"{content}\n"
            return md_text, None
