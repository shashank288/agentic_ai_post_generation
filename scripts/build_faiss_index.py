"""
Script to build FAISS index from documents in data/papers/.

Usage:
    python scripts/build_faiss_index.py

This script:
1. Reads all .txt files from data/papers/
2. Chunks them into smaller segments
3. Embeds them using Azure OpenAI
4. Builds and saves FAISS index to data/faiss_index/
"""

import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.retriever import FaissRetriever, Document

# Load environment variables
load_dotenv()


def load_documents_from_directory(directory: str) -> List[Document]:
    """
    Load all .txt files from a directory.
    
    Args:
        directory (str): Path to directory containing documents
        
    Returns:
        List[Document]: Loaded documents with metadata
    """
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory not found: {directory}")
        return documents
    
    # Find all .txt files
    txt_files = list(dir_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {directory}")
        return documents
    
    print(f"Found {len(txt_files)} documents")
    
    for file_path in txt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            documents.append(
                Document(
                    content=content,
                    metadata={
                        "source": file_path.name,
                        "path": str(file_path),
                    }
                )
            )
            print(f"Loaded: {file_path.name} ({len(content)} chars)")
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
    
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents (List[Document]): Documents to chunk
        chunk_size (int): Target chunk size in characters
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[Document]: Chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunked_docs = []
    
    for doc in documents:
        # Split text into chunks
        chunks = text_splitter.split_text(doc.content)
        
        # Create Document objects for each chunk
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
            )
    
    return chunked_docs


def main():
    """Build FAISS index from documents."""
    print("\n" + "="*60)
    print("BUILDING FAISS INDEX")
    print("="*60 + "\n")
    
    # Configuration
    papers_dir = os.getenv("PAPERS_DIR", "./data/papers")
    index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    print(f"Papers directory: {papers_dir}")
    print(f"Index path: {index_path}")
    print(f"Chunk size: {chunk_size} (overlap: {chunk_overlap})")
    print()
    
    # Step 1: Load documents
    print("Step 1: Loading documents...")
    documents = load_documents_from_directory(papers_dir)
    
    if not documents:
        print("\n No documents to index. Please add .txt files to data/papers/")
        return
    
    print(f"Loaded {len(documents)} documents\n")
    
    # Step 2: Chunk documents
    print("Step 2: Chunking documents...")
    chunked_docs = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunked_docs)} chunks\n")
    
    # Step 3: Initialize embeddings
    print("Step 3: Initializing Azure OpenAI embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    print("Embeddings initialized\n")
    
    # Step 4: Build FAISS index
    print("Step 4: Building FAISS index (this may take a few minutes)...")
    retriever = FaissRetriever(embeddings=embeddings)
    retriever.add_documents(chunked_docs)
    print(f"Index built with {len(chunked_docs)} vectors\n")
    
    # Step 5: Save index
    print(f"Step 5: Saving index to {index_path}...")
    retriever.save(index_path)
    
    # Step 6: Verify
    print("\nStep 6: Verifying index...")
    test_retriever = FaissRetriever(index_path=index_path)
    stats = test_retriever.get_stats()
    print(f"Index verified:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Vectors: {stats['total_vectors']}")
    print(f"   Dimension: {stats['dimension']}")
    
    print("\n" + "="*60)
    print("✅ FAISS INDEX BUILD COMPLETE")
    print("="*60)
    print(f"\nIndex saved to: {index_path}")
    print("You can now start the API and generate posts!\n")


if __name__ == "__main__":
    main()
