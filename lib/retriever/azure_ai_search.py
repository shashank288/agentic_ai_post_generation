"""
Azure AI Search retriever implementation (stub for future migration).

This module provides a stub implementation of the RetrieverBase interface
for Azure AI Search with hybrid (vector + keyword) search capabilities.

TODO: Implement this when migrating from FAISS to Azure AI Search.

Azure AI Search offers:
- Hybrid search (vector + BM25 keyword search)
- Managed service (no need to maintain indexes)
- Scalability and high availability
- Advanced features (facets, filters, geo-search)
"""

from typing import List

from .base import Document, RetrieverBase


class AzureAISearchRetriever(RetrieverBase):
    """
    Azure AI Search retriever with hybrid vector + keyword search.
    
    This is currently a stub implementation. When implementing:
    
    1. Install Azure SDK: pip install azure-search-documents
    2. Set up environment variables:
       - AZURE_SEARCH_ENDPOINT
       - AZURE_SEARCH_API_KEY
       - AZURE_SEARCH_INDEX_NAME
    3. Create search index with vector field configuration
    4. Implement hybrid query with vector + keyword search
    
    Example implementation outline:
    ```python
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    
    def search(self, query: str, k: int = 5):
        # Embed query
        query_vector = self.embeddings.embed_query(query)
        
        # Create hybrid query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k,
            fields="content_vector"
        )
        
        # Execute search with both vector and keyword
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["content", "metadata"],
            top=k
        )
        
        return [self._convert_to_document(r) for r in results]
    ```
    
    Attributes:
        search_client: Azure Search client instance
        embeddings: Azure OpenAI embeddings model
        index_name: Name of the search index
    """
    
    def __init__(self):
        """
        Initialize the Azure AI Search retriever.
        
        TODO: Implement initialization logic:
        - Load credentials from environment
        - Initialize SearchClient
        - Initialize embeddings model
        """
        raise NotImplementedError(
            "Azure AI Search retriever is not yet implemented. "
            "Use FaissRetriever for now. "
            "See azure_ai_search.py for implementation guidelines."
        )
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search using hybrid vector + keyword query.
        
        TODO: Implement hybrid search combining:
        - Vector similarity search on embeddings
        - BM25 keyword search on text fields
        - Reciprocal Rank Fusion (RRF) to merge results
        
        Args:
            query (str): Search query text
            k (int): Number of results to return
            
        Returns:
            List[Document]: Top-k results from hybrid search
        """
        raise NotImplementedError("Azure AI Search not yet implemented")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the Azure AI Search index.
        
        TODO: Implement document indexing:
        - Embed document content
        - Upload documents with vectors to Azure Search
        - Handle batching for large document sets
        
        Args:
            documents (List[Document]): Documents to index
        """
        raise NotImplementedError("Azure AI Search not yet implemented")
    
    def create_index(self) -> None:
        """
        Create the search index with vector configuration.
        
        TODO: Implement index creation with:
        - Text fields (content, title, etc.)
        - Vector field (content_vector with proper dimensions)
        - Metadata fields
        - Vector search configuration (algorithm, distance metric)
        """
        raise NotImplementedError("Azure AI Search not yet implemented")
    
    def delete_index(self) -> None:
        """Delete the search index."""
        raise NotImplementedError("Azure AI Search not yet implemented")


# Migration guide for switching from FAISS to Azure AI Search:
# 
# 1. Set up Azure AI Search resource in Azure Portal
# 2. Create search index with vector field (see create_index TODO)
# 3. Implement the methods above using azure-search-documents SDK
# 4. Update environment variables in .env
# 5. In your code, change:
#    from lib.retriever.faiss_store import FaissRetriever
#    to:
#    from lib.retriever.azure_ai_search import AzureAISearchRetriever
# 6. The interface is identical, so no other code changes needed!
