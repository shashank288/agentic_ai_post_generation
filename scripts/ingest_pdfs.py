"""
Production script to ingest PDFs into FAISS index.

Usage:
    python scripts/ingest_pdfs.py

Features:
- Azure Document Intelligence extraction
- GPT-4o figure captioning
- Configurable chunking strategies
- FAISS index persistence

Environment variables:
    AZURE_DI_ENDPOINT: Document Intelligence endpoint
    AZURE_DI_KEY: Document Intelligence key
    AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint (for GPT-4o)
    AZURE_OPENAI_API_KEY: Azure OpenAI key
    AZURE_OPENAI_CHAT_DEPLOYMENT: GPT-4o deployment (default: gpt-4o)
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT: Embeddings deployment
    AZURE_OPENAI_EMBEDDINGS_ENDPOINT: Embeddings endpoint
    AZURE_OPENAI_EMBEDDINGS_API_KEY: Embeddings key
    PDF_INPUT_DIR: Directory containing PDFs (default: ./data/papers)
    FAISS_INDEX_PATH: FAISS index output path (default: ./data/faiss_index)
    CHUNKING_STRATEGY: semantic | fixed_size | hybrid | atomic_elements (default: hybrid)
    CHUNK_SIZE: Target chunk size (default: 1000)
    CHUNK_OVERLAP: Chunk overlap (default: 200)
    CAPTION_FIGURES: Enable figure captioning (default: true)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.ingestion import (
    PDFExtractor,
    MarkdownNormalizer,
    DocumentChunker,
    ChunkingStrategy,
)
from lib.ingestion.chunker import ChunkingConfig
from lib.retriever import FaissRetriever

# Load environment
load_dotenv()


def get_config():
    """Load configuration from environment."""
    return {
        "pdf_input_dir": os.getenv("PDF_INPUT_DIR", "./data/papers"),
        "faiss_index_path": os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"),
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "hybrid"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
        "caption_figures": os.getenv("CAPTION_FIGURES", "true").lower() == "true",
        "extract_figures": os.getenv("EXTRACT_FIGURES", "true").lower() == "true",
        # Embedding batch settings (to handle rate limits)
        "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
        "embedding_batch_delay": float(os.getenv("EMBEDDING_BATCH_DELAY", "2.0")),
        "embedding_max_retries": int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
    }


def main():
    """Run the PDF ingestion pipeline."""
    print("\n" + "=" * 70)
    print("MULTIMODAL PDF INGESTION PIPELINE")
    print("=" * 70)
    
    # Load config
    config = get_config()
    pdf_dir = Path(config["pdf_input_dir"])
    index_path = config["faiss_index_path"]
    
    print(f"\n📂 Configuration:")
    print(f"   PDF input dir: {pdf_dir}")
    print(f"   FAISS index path: {index_path}")
    print(f"   Chunking strategy: {config['chunking_strategy']}")
    print(f"   Chunk size: {config['chunk_size']} (overlap: {config['chunk_overlap']})")
    print(f"   Figure captioning: {config['caption_figures']}")
    
    # Find PDFs
    if not pdf_dir.exists():
        print(f"\n❌ Error: PDF directory not found: {pdf_dir}")
        print(f"   Please create it and add PDF files.")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in {pdf_dir}")
        print(f"   Please add PDF files to ingest.")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF(s) to process")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    # 1. PDF Extractor
    extractor = PDFExtractor(caption_figures=config["caption_figures"])
    print("   ✓ PDF Extractor (Azure Document Intelligence + GPT-4o)")
    
    # 2. Normalizer
    normalizer = MarkdownNormalizer(include_page_markers=True)
    print("   ✓ Markdown Normalizer")
    
    # 3. Chunker
    chunking_strategy = ChunkingStrategy(config["chunking_strategy"])
    chunking_config = ChunkingConfig(
        strategy=chunking_strategy,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    chunker = DocumentChunker(config=chunking_config)
    print(f"   ✓ Document Chunker ({chunking_strategy.value})")
    
    # 4. Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    print("   ✓ Azure OpenAI Embeddings")
    
    # 5. FAISS Retriever
    retriever = FaissRetriever(embeddings=embeddings)
    print("   ✓ FAISS Retriever")
    
    # Process PDFs
    print("\n" + "=" * 70)
    print("PROCESSING PDFs")
    print("=" * 70)
    
    all_chunks = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
        print("-" * 70)
        
        try:
            # Step 1: Extract
            extracted = extractor.extract(
                str(pdf_path),
                extract_figures=config["extract_figures"],
            )
            
            # Step 2: Normalize
            normalized = normalizer.normalize(extracted)
            
            # Step 3: Chunk
            chunks = chunker.chunk(normalized)
            
            all_chunks.extend(chunks)
            
            print(f"  ✅ Processed successfully ({len(chunks)} chunks)")
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_chunks:
        print("\n❌ No chunks created. Please check PDF processing errors.")
        return
    
    # Build FAISS index
    print("\n" + "=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)
    
    print(f"\n📊 Total chunks: {len(all_chunks)}")
    print(f"⏳ Embedding {len(all_chunks)} chunks in batches of {config['embedding_batch_size']}...")
    print(f"   (Delay: {config['embedding_batch_delay']}s between batches, Max retries: {config['embedding_max_retries']})\n")
    
    retriever.add_documents(
        all_chunks,
        batch_size=config["embedding_batch_size"],
        delay_between_batches=config["embedding_batch_delay"],
        max_retries=config["embedding_max_retries"],
    )
    
    print(f"✅ Index built with {len(all_chunks)} vectors")
    
    # Save index
    print(f"\n💾 Saving index to {index_path}...")
    retriever.save(index_path)
    
    # Verify
    print("\n🔍 Verifying index...")
    test_retriever = FaissRetriever(index_path=index_path)
    stats = test_retriever.get_stats()
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Vectors: {stats['total_vectors']}")
    print(f"   Dimension: {stats['dimension']}")
    
    print("\n" + "=" * 70)
    print("✅ INGESTION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Index saved to: {index_path}")
    print(f"📈 Total documents indexed: {stats['total_documents']}")
    print("\n💡 Next steps:")
    print("   1. Start the API: uvicorn api.main:app --reload")
    print("   2. Test retrieval in notebooks/04_retrieval.ipynb")
    print("   3. Generate posts with multimodal context!\n")


if __name__ == "__main__":
    main()
