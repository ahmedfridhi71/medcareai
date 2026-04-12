"""
ChromaDB Vector Store Module.

Manages document storage and retrieval using ChromaDB.
Stores document chunks with embeddings and metadata for semantic search.
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from rag.chunker import DocumentChunk
from rag.embedder import DocumentEmbedder, get_embedder


class VectorStore:
    """
    Vector store for medical documents using ChromaDB.
    
    Stores document chunks with embeddings for semantic retrieval.
    """
    
    COLLECTION_NAME = "medcareai_docs"
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedder: Optional[DocumentEmbedder] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data.
                              If None, uses in-memory storage.
            embedder: DocumentEmbedder instance. Creates default if None.
        """
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
        
        # Initialize embedder
        self.embedder = embedder or get_embedder()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "MedCareAI medical documents"}
        )
    
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to process at once
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Process in batches
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in batch]
            documents = [chunk.content for chunk in batch]
            
            # Create embeddings
            embeddings = self.embedder.embed_texts(
                documents,
                show_progress=len(documents) > 10
            )
            
            # Prepare metadata
            metadatas = [
                {
                    "source": chunk.source,
                    "title": chunk.title,
                    "url": chunk.url,
                    "disease": chunk.disease or "",
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks
                }
                for chunk in batch
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            print(f"Added {total_added}/{len(chunks)} chunks...")
        
        return total_added
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        disease_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: Query text (disease name, symptoms, or question)
            n_results: Number of results to return
            disease_filter: Optional filter by disease name
            
        Returns:
            List of results with content, metadata, and distance
        """
        # Create query embedding
        query_embedding = self.embedder.embed_query(query_text)
        
        # Build where filter if disease specified
        where_filter = None
        if disease_filter:
            where_filter = {"disease": {"$eq": disease_filter.lower()}}
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "relevance_score": 1 - (results["distances"][0][i] if results["distances"] else 0)
                })
        
        return formatted_results
    
    def query_by_disease(
        self,
        disease_name: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Query specifically for a disease.
        
        Args:
            disease_name: Name of the disease
            n_results: Number of results
            
        Returns:
            List of relevant document chunks
        """
        # Use disease name as query
        query = f"What is {disease_name}? Symptoms, causes, treatment of {disease_name}"
        return self.query(query, n_results=n_results)
    
    def get_all_diseases(self) -> List[str]:
        """
        Get list of all diseases in the vector store.
        
        Returns:
            List of unique disease names
        """
        # Get all metadata
        results = self.collection.get(include=["metadatas"])
        
        diseases = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta.get("disease"):
                    diseases.add(meta["disease"])
        
        return sorted(list(diseases))
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "MedCareAI medical documents"}
        )
    
    def delete_by_disease(self, disease_name: str) -> int:
        """
        Delete all chunks for a specific disease.
        
        Args:
            disease_name: Disease name to delete
            
        Returns:
            Number of chunks deleted
        """
        # Get IDs for this disease
        results = self.collection.get(
            where={"disease": {"$eq": disease_name.lower()}},
            include=[]
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        
        return 0


def get_vector_store(
    persist_directory: Optional[str] = None
) -> VectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        persist_directory: Directory for persistence (optional)
        
    Returns:
        Configured VectorStore instance
    """
    return VectorStore(persist_directory=persist_directory)
