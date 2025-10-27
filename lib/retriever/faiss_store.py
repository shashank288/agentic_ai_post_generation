"""
FAISS-based retriever implementation.

This module implements the RetrieverBase interface using FAISS for efficient
vector similarity search. It supports saving/loading indexes to disk and
uses Azure OpenAI embeddings.

FAISS is an in-memory vector database that provides fast similarity search.
Indexes are persisted to disk to avoid re-embedding documents on every startup.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings

from .base import Document, RetrieverBase


class FaissRetriever(RetrieverBase):
    """
    FAISS-based document retriever using vector similarity search.
    
    This implementation:
    - Embeds documents and queries using Azure OpenAI embeddings
    - Stores vectors in a FAISS index for efficient similarity search
    - Supports save/load operations to persist the index to disk
    - Returns top-k most similar documents for a given query
    
    Attributes:
        embeddings (AzureOpenAIEmbeddings): Azure OpenAI embeddings model
        index (faiss.Index): FAISS index for vector search
        documents (List[Document]): List of stored documents
        dimension (int): Embedding vector dimension
    """
    
    def __init__(
        self,
        embeddings: Optional[AzureOpenAIEmbeddings] = None,
        index_path: Optional[str] = None,
    ):
        """
        Initialize the FAISS retriever.
        
        Args:
            embeddings (Optional[AzureOpenAIEmbeddings]): Embeddings model instance.
                If None, will be created from environment variables.
            index_path (Optional[str]): Path to load/save FAISS index.
                If provided and exists, loads the index from disk.
        """
        # Initialize embeddings model
        if embeddings is None:
            # Allow separate endpoint/key for embeddings; fallback to main Azure OpenAI envs
            embeddings_endpoint = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
            embeddings_api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            embeddings_api_version = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")

            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
                azure_endpoint=embeddings_endpoint,
                api_key=embeddings_api_key,
                openai_api_version=embeddings_api_version,
            )
        else:
            self.embeddings = embeddings
        
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.dimension: int = 1536  # text-embedding-ada-002 dimension
        self.index_path = index_path
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            # Initialize empty index
            self._initialize_index()
    
    def _initialize_index(self) -> None:
        """
        Initialize an empty FAISS index.
        
        Creates a new IndexFlatL2 index for L2 (Euclidean) distance similarity.
        This is suitable for small to medium-sized datasets (~thousands of docs).
        
        For larger datasets, consider IndexIVFFlat or IndexHNSW.
        """
        # Use L2 distance (Euclidean) for similarity
        # For cosine similarity, normalize vectors before adding
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query (str): Search query text
            k (int): Number of top results to return (default: 5)
            
        Returns:
            List[Document]: Top-k most similar documents with relevance scores
            
        Raises:
            ValueError: If index is empty or not initialized
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. Add documents before searching.")
        
        # Embed the query
        query_vector = self.embeddings.embed_query(query)
        query_array = np.array([query_vector], dtype=np.float32)
        
        # Search the index
        # distances: lower is better (L2 distance)
        # indices: positions in the documents list
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Build result documents
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            doc = self.documents[idx]
            # Convert L2 distance to similarity score (higher = more similar)
            # Using inverse distance: score = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + float(dist))
            
            results.append(
                Document(
                    content=doc.content,
                    metadata={**doc.metadata, "distance": float(dist)},
                    score=similarity_score,
                )
            )
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the FAISS index.
        
        Args:
            documents (List[Document]): Documents to add
            
        Notes:
            - Embeds all documents using the embeddings model
            - Adds embeddings to the FAISS index
            - Stores documents for later retrieval
        """
        if not documents:
            return
        
        # Extract text content
        texts = [doc.content for doc in documents]
        
        # Embed all documents
        embeddings = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        if self.index is None:
            self._initialize_index()
        
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents.extend(documents)
    
    def save(self, path: str) -> None:
        """
        Save the FAISS index and documents to disk.
        
        Args:
            path (str): Directory path to save index and metadata
            
        Notes:
            Saves two files:
            - index.faiss: The FAISS index
            - documents.pkl: Pickled list of documents
        """
        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save documents metadata
        docs_file = os.path.join(path, "documents.pkl")
        with open(docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ FAISS index saved to {path} ({self.index.ntotal} vectors)")
    
    def load(self, path: str) -> None:
        """
        Load the FAISS index and documents from disk.
        
        Args:
            path (str): Directory path containing saved index and metadata
            
        Raises:
            FileNotFoundError: If index files don't exist
        """
        index_file = os.path.join(path, "index.faiss")
        docs_file = os.path.join(path, "documents.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            raise FileNotFoundError(f"Index files not found at {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load documents
        with open(docs_file, "rb") as f:
            self.documents = pickle.load(f)
        
        print(f"✅ FAISS index loaded from {path} ({self.index.ntotal} vectors)")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            dict: Statistics including document count, index size, etc.
        """
        return {
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None,
        }
