# Multimodal PDF Ingestion Pipeline

Complete guide to the robust PDF ingestion pipeline using Azure Document Intelligence and GPT-4o.

## Overview

The ingestion pipeline extracts, normalizes, and indexes multimodal PDF documents (text, tables, figures) into a FAISS vector database for retrieval-augmented generation.

### Architecture

```
PDFs → Extraction → Normalization → Chunking → Embedding → FAISS Index
         (Azure DI)    (Markdown)   (Strategies)  (Azure)    (Vector DB)
         (GPT-4o)
```

### Key Features

- **Layout-aware extraction**: Preserves document structure (headings, sections, tables)
- **Table extraction**: Maintains table structure in Markdown format
- **Figure captioning**: GPT-4o generates descriptive captions for images/figures
- **Configurable chunking**: 4 strategies (semantic, fixed-size, hybrid, atomic)
- **Production-ready**: Modular design with scripts + experimental notebooks

---

## Components

### 1. PDF Extractor (`lib/ingestion/extractor.py`)

Extracts structured content from PDFs using Azure Document Intelligence.

**Features:**
- Text blocks with layout (paragraphs, headings, page numbers)
- Tables with structure preserved
- Figures with bounding boxes
- Optional GPT-4o figure captioning

**Usage:**
```python
from lib.ingestion import PDFExtractor

extractor = PDFExtractor(caption_figures=True)
extracted = extractor.extract("path/to/paper.pdf")

# Access extracted content
print(f"Text blocks: {len(extracted.text_blocks)}")
print(f"Tables: {len(extracted.tables)}")
print(f"Figures: {len(extracted.figures)}")
```

### 2. Markdown Normalizer (`lib/ingestion/normalizer.py`)

Converts extracted content to structured Markdown.

**Features:**
- Unified timeline (merges text, tables, figures by page/position)
- Markdown formatting (headings, paragraphs, tables, captions)
- Page markers (optional)
- Section metadata

**Usage:**
```python
from lib.ingestion import MarkdownNormalizer

normalizer = MarkdownNormalizer(include_page_markers=True)
normalized = normalizer.normalize(extracted)

# Access normalized content
print(normalized.content)  # Full Markdown text
print(f"Sections: {len(normalized.sections)}")
```

### 3. Document Chunker (`lib/ingestion/chunker.py`)

Splits normalized documents into chunks for embedding.

**Chunking Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SEMANTIC` | Chunk by headings/sections | Preserve document structure |
| `FIXED_SIZE` | Fixed char count + overlap | Uniform chunk sizes |
| `HYBRID` | Semantic + split large sections | **Recommended** - balance structure & size |
| `ATOMIC_ELEMENTS` | Keep tables/figures as single chunks | Preserve table/figure integrity |

**Configuration:**
```python
from lib.ingestion import DocumentChunker, ChunkingStrategy
from lib.ingestion.chunker import ChunkingConfig

config = ChunkingConfig(
    strategy=ChunkingStrategy.HYBRID,
    chunk_size=1000,          # Target size in chars
    chunk_overlap=200,        # Overlap between chunks
    max_chunk_size=2500,      # Max before splitting
    keep_tables_atomic=True,  # Don't split tables
    keep_figures_atomic=True, # Don't split figures
)

chunker = DocumentChunker(config=config)
chunks = chunker.chunk(normalized)
```

---

## Production Usage

### Setup

1. **Install dependencies:**
   ```bash
   pip install azure-ai-documentintelligence pillow pymupdf
   ```

2. **Configure environment variables** (`.env`):
   ```bash
   # Azure Document Intelligence
   AZURE_DI_ENDPOINT=https://your-di-resource.cognitiveservices.azure.com/
   AZURE_DI_KEY=your-di-key

   # Azure OpenAI (for GPT-4o captioning + embeddings)
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small

   # Ingestion configuration
   PDF_INPUT_DIR=./data/papers
   FAISS_INDEX_PATH=./data/faiss_index
   CHUNKING_STRATEGY=hybrid
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   CAPTION_FIGURES=true
   ```

3. **Add PDFs to input directory:**
   ```bash
   mkdir -p data/papers
   # Copy your PDF files to data/papers/
   ```

### Run Ingestion

```bash
python scripts/ingest_pdfs.py
```

**Output:**
- FAISS index: `./data/faiss_index/`
- Files: `index.faiss`, `documents.pkl`

**Expected log:**
```
MULTIMODAL PDF INGESTION PIPELINE
======================================================================

📂 Configuration:
   PDF input dir: data/papers
   FAISS index path: data/faiss_index
   Chunking strategy: hybrid
   Chunk size: 1000 (overlap: 200)
   Figure captioning: True

📚 Found 1 PDF(s) to process

🔧 Initializing components...
   ✓ PDF Extractor (Azure Document Intelligence + GPT-4o)
   ✓ Markdown Normalizer
   ✓ Document Chunker (hybrid)
   ✓ Azure OpenAI Embeddings
   ✓ FAISS Retriever

PROCESSING PDFs
======================================================================

[1/1] 2401.13138v6.pdf
----------------------------------------------------------------------

📄 Extracting: 2401.13138v6.pdf
  ⏳ Running Azure Document Intelligence Layout analysis...
  ✓ Analysis complete: 14 pages
  ✓ Extracted 87 text blocks
  ✓ Extracted 3 tables
  ✓ Extracted and captioned 5 figures

📝 Normalizing to Markdown...
  ✓ Normalized 95 sections (24531 chars)

✂️  Chunking with strategy: hybrid
  ✓ Created 32 chunks

  ✅ Processed successfully (32 chunks)

BUILDING FAISS INDEX
======================================================================

📊 Total chunks: 32
⏳ Embedding 32 chunks (this may take a few minutes)...
✅ Index built with 32 vectors

💾 Saving index to data/faiss_index...
✅ FAISS index saved to data/faiss_index (32 vectors)

✅ INGESTION COMPLETE
======================================================================

📁 Index saved to: data/faiss_index
📈 Total documents indexed: 32
```

---

## Experimental Notebooks

Four Jupyter notebooks for experimentation and debugging:

### 1. `notebooks/01_extraction_normalization.ipynb`
- Test extraction with Azure Document Intelligence
- Inspect text blocks, tables, figures
- View GPT-4o captions
- Export Markdown for manual inspection

### 2. `notebooks/02_chunking_experiments.ipynb`
- Compare all 4 chunking strategies
- Visualize chunk size distributions
- Tune chunk_size and overlap parameters
- Analyze chunk content types

### 3. `notebooks/03_embedding_and_index.ipynb`
- Test Azure OpenAI embeddings
- Build FAISS index
- Benchmark embedding speed
- Save/load index from disk

### 4. `notebooks/04_retrieval_testing.ipynb`
- Test retrieval queries
- Evaluate relevance
- Test multimodal retrieval (tables, figures)
- Analyze retrieval diversity

**Run notebooks:**
```bash
jupyter notebook notebooks/
```

---

## Configuration Guide

### Chunking Strategy Selection

**SEMANTIC** (`semantic`):
- ✅ Preserves document structure
- ✅ Natural section boundaries
- ❌ Variable chunk sizes (can be very large)
- **Use for**: Well-structured documents with clear headings

**FIXED_SIZE** (`fixed_size`):
- ✅ Uniform chunk sizes
- ✅ Predictable token counts
- ❌ Ignores document structure (may split mid-sentence)
- **Use for**: When consistency is more important than structure

**HYBRID** (`hybrid`) - **Recommended**:
- ✅ Best of both: structure + size control
- ✅ Splits large sections automatically
- ✅ Respects semantic boundaries when possible
- **Use for**: Most research papers and technical documents

**ATOMIC_ELEMENTS** (`atomic_elements`):
- ✅ Tables/figures never split
- ✅ Preserves multimodal element integrity
- ❌ More complex chunking logic
- **Use for**: Documents with many tables/figures that must remain intact

### Parameter Tuning

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `chunk_size` | 800-1200 | Smaller = more precise, larger = more context |
| `chunk_overlap` | 150-250 | ~15-20% of chunk_size |
| `max_chunk_size` | 2500-3000 | For HYBRID strategy |
| `caption_figures` | `true` | Enables GPT-4o figure captioning (adds cost) |

**Example configurations:**

```bash
# High precision (small chunks)
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# Balanced (recommended)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# More context (larger chunks)
CHUNK_SIZE=1500
CHUNK_OVERLAP=250
```

---

## Cost Estimation

### Azure Document Intelligence
- **Layout model**: ~$10 per 1,000 pages
- **Example**: 100-page document = $1.00

### GPT-4o (Figure Captioning)
- **Price**: ~$2.50 per 1M input tokens
- **Avg figure**: ~1,000 tokens (image encoding) + 150 tokens (response) = 1,150 tokens
- **Cost per figure**: ~$0.003
- **Example**: 20 figures = $0.06

### Azure OpenAI Embeddings
- **text-embedding-3-small**: $0.02 per 1M tokens
- **Avg chunk**: ~200 tokens
- **Example**: 100 chunks = 20K tokens = $0.0004

**Total for 100-page paper with 20 figures and 100 chunks**: ~$1.06

---

## Troubleshooting

### Issue: "Azure Document Intelligence credentials required"
**Solution**: Set `AZURE_DI_ENDPOINT` and `AZURE_DI_KEY` in `.env`

### Issue: Figure captioning fails
**Solution**: 
- Check `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
- Verify `AZURE_OPENAI_CHAT_DEPLOYMENT` points to a vision-capable model (gpt-4o, gpt-4-vision-preview)
- Set `CAPTION_FIGURES=false` to disable captioning

### Issue: Chunks too large/small
**Solution**: Adjust `CHUNK_SIZE` and `MAX_CHUNK_SIZE` in `.env` or experiment in notebook 02

### Issue: Tables are split incorrectly
**Solution**: Use `ATOMIC_ELEMENTS` strategy with `keep_tables_atomic=True`

### Issue: Embedding too slow
**Solution**: 
- Reduce number of chunks (increase `CHUNK_SIZE`)
- Azure OpenAI embeddings are rate-limited; consider batching or retry logic

---

## Advanced Usage

### Custom Extraction Pipeline

```python
from lib.ingestion import PDFExtractor, MarkdownNormalizer, DocumentChunker
from lib.ingestion.chunker import ChunkingConfig, ChunkingStrategy

# 1. Extract with custom settings
extractor = PDFExtractor(
    caption_figures=True,
    di_endpoint="...",
    di_key="...",
)
extracted = extractor.extract("paper.pdf", extract_figures=True)

# 2. Normalize
normalizer = MarkdownNormalizer(include_page_markers=False)
normalized = normalizer.normalize(extracted)

# 3. Chunk with custom config
config = ChunkingConfig(
    strategy=ChunkingStrategy.ATOMIC_ELEMENTS,
    chunk_size=1200,
    chunk_overlap=250,
    keep_tables_atomic=True,
)
chunker = DocumentChunker(config=config)
chunks = chunker.chunk(normalized)

# 4. Build index
from lib.retriever import FaissRetriever
retriever = FaissRetriever(embeddings=your_embeddings)
retriever.add_documents(chunks)
retriever.save("./my_index")
```

### Batch Processing Multiple PDFs

```python
from pathlib import Path

pdf_dir = Path("./data/papers")
all_chunks = []

for pdf_path in pdf_dir.glob("*.pdf"):
    extracted = extractor.extract(str(pdf_path))
    normalized = normalizer.normalize(extracted)
    chunks = chunker.chunk(normalized)
    all_chunks.extend(chunks)

# Build single index from all PDFs
retriever.add_documents(all_chunks)
retriever.save("./multi_doc_index")
```

---

## Next Steps

1. **Run ingestion**: `python scripts/ingest_pdfs.py`
2. **Test retrieval**: Open `notebooks/04_retrieval_testing.ipynb`
3. **Integrate with API**: The FAISS index at `./data/faiss_index` is ready for use with the existing retrieval system
4. **Generate posts**: Run the API and generate posts with multimodal context!

---

## References

- [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [Azure OpenAI GPT-4o](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4o-and-gpt-4-turbo)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
