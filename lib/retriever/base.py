"""
Base retriever interface for document retrieval.

This module defines the abstract base class for all retriever implementations.
Implementations must provide a `search` method that returns relevant documents.

The interface allows for easy swapping between FAISS, Azure AI Search, or other
retrieval backends without changing downstream code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """
    Represents a retrieved document with content and metadata.
    
    Attributes:
        content (str): The text content of the document chunk
        metadata (dict): Additional metadata (source, page, score, etc.)
        score (float): Relevance score from the retriever (optional, higher = more relevant)
    """
    content: str
    metadata: dict
    score: float = 0.0


class RetrieverBase(ABC):
    """
    Abstract base class for document retrieval systems.
    
    All retriever implementations (FAISS, Azure AI Search, etc.) must inherit
    from this class and implement the `search` method.
    
    This abstraction allows the application to switch between different
    retrieval backends with minimal code changes.
    """
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents given a query.
        
        Args:
            query (str): The search query text
            k (int): Number of top documents to return (default: 5)
            
        Returns:
            List[Document]: List of retrieved documents sorted by relevance
            
        Raises:
            NotImplementedError: If the method is not implemented in subclass
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retrieval index.
        
        Args:
            documents (List[Document]): Documents to add to the index
            
        Raises:
            NotImplementedError: If the method is not implemented in subclass
        """
        pass
    
    def health_check(self) -> bool:
        """
        Check if the retriever is operational.
        
        Returns:
            bool: True if retriever is healthy, False otherwise
        """
        try:
            # Attempt a dummy search to verify the retriever works
            self.search("health check", k=1)
            return True
        except Exception:
            return False
