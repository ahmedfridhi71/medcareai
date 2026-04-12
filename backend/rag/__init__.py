"""
MedCareAI RAG Package.

Contains document embedding, retrieval, and ChromaDB integration for
evidence-based medical explanations.
"""

from rag.chunker import DocumentChunk, process_document, process_documents_directory
from rag.embedder import DocumentEmbedder, get_embedder
from rag.vector_store import VectorStore, get_vector_store
from rag.rag_service import generate_explanation, quick_answer, get_available_topics

__all__ = [
    "DocumentChunk",
    "process_document",
    "process_documents_directory",
    "DocumentEmbedder",
    "get_embedder",
    "VectorStore",
    "get_vector_store",
    "generate_explanation",
    "quick_answer",
    "get_available_topics",
]
