"""
PDF extraction using Azure Document Intelligence and GPT-4o.

Extracts:
- Text with layout (paragraphs, headings)
- Tables (with structure preserved)
- Figures (with GPT-4o-generated captions)
- Page/section metadata
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


@dataclass
class ExtractedContent:
    """Container for extracted PDF content."""
    text_blocks: List[Dict]  # paragraphs/headings with position/page
    tables: List[Dict]  # table data with markdown + position/page
    figures: List[Dict]  # figure crops with GPT-4o caption + position/page
    metadata: Dict  # document-level metadata


class PDFExtractor:
    """
    Extract structured content from PDFs using Azure Document Intelligence.
    
    Features:
    - Layout analysis with headings/paragraphs/tables
    - Table structure extraction
    - Figure detection and cropping
    - Optional GPT-4o figure captioning
    """
    
    def __init__(
        self,
        di_endpoint: Optional[str] = None,
        di_key: Optional[str] = None,
        openai_endpoint: Optional[str] = None,
        openai_key: Optional[str] = None,
        openai_deployment: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        caption_figures: bool = True,
    ):
        """
        Initialize PDF extractor.
        
        Args:
            di_endpoint: Azure Document Intelligence endpoint
            di_key: Azure Document Intelligence key
            openai_endpoint: Azure OpenAI endpoint for GPT-4o
            openai_key: Azure OpenAI key
            openai_deployment: GPT-4o deployment name
            openai_api_version: Azure OpenAI API version
            caption_figures: Whether to generate captions with GPT-4o
        """
        # Document Intelligence client
        self.di_endpoint = di_endpoint or os.getenv("AZURE_DI_ENDPOINT")
        self.di_key = di_key or os.getenv("AZURE_DI_KEY")
        
        if not self.di_endpoint or not self.di_key:
            raise ValueError("Azure Document Intelligence credentials required (AZURE_DI_ENDPOINT, AZURE_DI_KEY)")
        
        self.di_client = DocumentIntelligenceClient(
            endpoint=self.di_endpoint,
            credential=AzureKeyCredential(self.di_key)
        )
        
        # GPT-4o client for figure captioning
        self.caption_figures = caption_figures
        if caption_figures:
            self.openai_endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.openai_key = openai_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.openai_deployment = openai_deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
            self.openai_api_version = openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
            
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.openai_endpoint,
                api_key=self.openai_key,
                api_version=self.openai_api_version,
            )
    
    def extract(self, pdf_path: str, extract_figures: bool = True) -> ExtractedContent:
        """
        Extract all content from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            extract_figures: Whether to extract and caption figures
            
        Returns:
            ExtractedContent with text, tables, figures
        """
        print(f"\n📄 Extracting: {Path(pdf_path).name}")
        
        # Step 1: Analyze with Document Intelligence
        print("  ⏳ Running Azure Document Intelligence Layout analysis...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        poller = self.di_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=pdf_bytes),
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()
        print(f"  ✓ Analysis complete: {len(result.pages)} pages")
        
        # Step 2: Extract text blocks (paragraphs, headings)
        text_blocks = self._extract_text_blocks(result)
        print(f"  ✓ Extracted {len(text_blocks)} text blocks")
        
        # Step 3: Extract tables
        tables = self._extract_tables(result)
        print(f"  ✓ Extracted {len(tables)} tables")
        
        # Step 4: Extract and caption figures
        figures = []
        if extract_figures and self.caption_figures:
            figures = self._extract_and_caption_figures(pdf_path, result)
            print(f"  ✓ Extracted and captioned {len(figures)} figures")
        
        # Metadata
        metadata = {
            "filename": Path(pdf_path).name,
            "num_pages": len(result.pages),
            "num_text_blocks": len(text_blocks),
            "num_tables": len(tables),
            "num_figures": len(figures),
        }
        
        return ExtractedContent(
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            metadata=metadata,
        )
    
    def _extract_text_blocks(self, result) -> List[Dict]:
        """Extract paragraphs and headings with position/page metadata."""
        blocks = []
        
        for paragraph in result.paragraphs:
            block = {
                "type": paragraph.role if paragraph.role else "paragraph",  # title, sectionHeading, paragraph
                "content": paragraph.content,
                "page": self._get_page_number(paragraph, result),
                "bounding_regions": paragraph.bounding_regions if paragraph.bounding_regions else [],
            }
            blocks.append(block)
        
        return blocks
    
    def _extract_tables(self, result) -> List[Dict]:
        """Extract tables with markdown representation."""
        tables = []
        
        if not result.tables:
            return tables
        
        for idx, table in enumerate(result.tables):
            # Build markdown table
            markdown_table = self._table_to_markdown(table)
            
            table_data = {
                "type": "table",
                "index": idx,
                "content": markdown_table,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "page": self._get_page_number(table, result),
                "bounding_regions": table.bounding_regions if table.bounding_regions else [],
            }
            tables.append(table_data)
        
        return tables
    
    def _table_to_markdown(self, table) -> str:
        """Convert DI table to markdown format."""
        if not table.cells:
            return ""
        
        # Determine table dimensions
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1
        
        # Build grid
        grid = [["" for _ in range(max_col)] for _ in range(max_row)]
        
        for cell in table.cells:
            content = cell.content.strip() if cell.content else ""
            grid[cell.row_index][cell.column_index] = content
        
        # Convert to markdown
        lines = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:  # header separator
                lines.append("| " + " | ".join(["---"] * len(row)) + " |")
        
        return "\n".join(lines)
    
    def _extract_and_caption_figures(self, pdf_path: str, result) -> List[Dict]:
        """Extract figure regions and generate captions with GPT-4o."""
        figures = []
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Find figure regions from DI result
        figure_regions = self._identify_figure_regions(result)
        
        for idx, region in enumerate(figure_regions):
            page_num = region["page"]
            bbox = region["bbox"]  # (x0, y0, x1, y1) in points
            
            # Crop figure from PDF page
            page = doc[page_num]
            pix = page.get_pixmap(clip=bbox, dpi=150)
            img_bytes = pix.tobytes("png")
            
            # Generate caption with GPT-4o
            caption = self._caption_image(img_bytes, page_num, idx)
            
            figures.append({
                "type": "figure",
                "index": idx,
                "page": page_num + 1,  # 1-indexed
                "bbox": bbox,
                "caption": caption,
                "image_bytes": img_bytes,  # optional: store for debugging
            })
        
        doc.close()
        return figures
    
    def _identify_figure_regions(self, result) -> List[Dict]:
        """
        Identify figure/image regions from DI result.
        
        Heuristic: look for figures in result.figures or large image bounding boxes.
        """
        regions = []
        
        # Method 1: Use result.figures if available (DI Layout can detect figures)
        if hasattr(result, "figures") and result.figures:
            for fig in result.figures:
                for region in fig.bounding_regions:
                    page_num = region.page_number - 1  # 0-indexed
                    polygon = region.polygon
                    # Convert polygon to bbox (x0, y0, x1, y1)
                    xs = [polygon[i] for i in range(0, len(polygon), 2)]
                    ys = [polygon[i] for i in range(1, len(polygon), 2)]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    regions.append({"page": page_num, "bbox": bbox})
        
        # Method 2: Fallback - detect large non-text regions (optional enhancement)
        # For now, rely on DI's figure detection
        
        return regions
    
    def _caption_image(self, img_bytes: bytes, page_num: int, fig_idx: int) -> str:
        """Generate caption for an image using GPT-4o vision."""
        # Encode image to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This is a figure from a research paper. "
                                    "Provide a concise caption (1-2 sentences) describing what this figure shows. "
                                    "Focus on the key information, data, or concept illustrated."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=150,
            )
            caption = response.choices[0].message.content.strip()
            return caption
        except Exception as e:
            print(f"    ⚠ Warning: Failed to caption figure {fig_idx} on page {page_num + 1}: {e}")
            return f"Figure {fig_idx + 1} (page {page_num + 1})"
    
    def _get_page_number(self, element, result) -> int:
        """Get 1-indexed page number from element's bounding regions."""
        if element.bounding_regions and len(element.bounding_regions) > 0:
            return element.bounding_regions[0].page_number
        return 1  # default to page 1 if unknown
