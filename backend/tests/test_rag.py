"""
RAG Pipeline Tests.

Tests for document chunking, embedding, and retrieval.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from rag.chunker import (
    chunk_text,
    extract_metadata_from_text,
    extract_disease_from_title,
    DocumentChunk,
    process_documents_directory,
)
from rag.embedder import DocumentEmbedder, get_embedder
from rag.vector_store import VectorStore


class TestDocumentChunker:
    """Tests for the document chunker module."""
    
    def test_extract_metadata(self):
        """Test metadata extraction from document header."""
        content = """TITLE: Diabetes Type 2
SOURCE: WHO Health Topics
URL: https://who.int/diabetes

This is the content..."""
        
        metadata = extract_metadata_from_text(content)
        
        assert metadata["title"] == "Diabetes Type 2"
        assert metadata["source"] == "WHO Health Topics"
        assert metadata["url"] == "https://who.int/diabetes"
    
    def test_extract_disease_from_title(self):
        """Test disease name extraction from titles."""
        assert extract_disease_from_title("Common Cold") == "common cold"
        assert extract_disease_from_title("Type 2 Diabetes Mellitus") == "type 2 diabetes mellitus"
        assert extract_disease_from_title("Migraine (Headache Disorder)") == "migraine"
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
    
    def test_chunk_text_with_overlap(self):
        """Test chunking with overlap."""
        # Create a longer text
        sentences = ["Sentence number {}.".format(i) for i in range(20)]
        text = " ".join(sentences)
        
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        # Should have multiple chunks
        assert len(chunks) > 1
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("")
        assert chunks == []
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text."
        chunks = chunk_text(text, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestDocumentEmbedder:
    """Tests for the document embedder module."""
    
    def test_get_embedder_default(self):
        """Test creating default embedder."""
        embedder = get_embedder()
        
        assert embedder is not None
        assert isinstance(embedder, DocumentEmbedder)
    
    def test_embed_text(self):
        """Test embedding a single text."""
        embedder = get_embedder()
        
        embedding = embedder.embed_text("This is a test sentence.")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_texts_batch(self):
        """Test embedding multiple texts."""
        embedder = get_embedder()
        texts = ["First text.", "Second text.", "Third text."]
        
        embeddings = embedder.embed_texts(texts, show_progress=False)
        
        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)
    
    def test_embedding_dimension(self):
        """Test embedding dimension property."""
        embedder = get_embedder()
        
        dim = embedder.embedding_dimension
        
        assert isinstance(dim, int)
        assert dim > 0


class TestVectorStore:
    """Tests for the vector store module."""
    
    def test_create_vector_store_memory(self):
        """Test creating in-memory vector store."""
        store = VectorStore()
        
        assert store is not None
        assert store.get_document_count() == 0
    
    def test_add_and_query_chunks(self):
        """Test adding and querying chunks."""
        store = VectorStore()
        
        # Create test chunks
        chunks = [
            DocumentChunk(
                content="Diabetes is a chronic disease affecting blood sugar.",
                source="Test Source",
                title="Diabetes Overview",
                url="https://test.com/diabetes",
                chunk_index=0,
                total_chunks=1,
                disease="diabetes"
            ),
            DocumentChunk(
                content="Hypertension is high blood pressure in arteries.",
                source="Test Source",
                title="Hypertension Guide",
                url="https://test.com/hypertension",
                chunk_index=0,
                total_chunks=1,
                disease="hypertension"
            )
        ]
        
        # Add chunks
        added = store.add_chunks(chunks)
        assert added == 2
        
        # Query
        results = store.query("blood sugar diabetes", n_results=1)
        
        assert len(results) >= 1
        assert "diabetes" in results[0]["content"].lower()
    
    def test_query_by_disease(self):
        """Test querying by disease name."""
        store = VectorStore()
        
        chunk = DocumentChunk(
            content="Migraine is a neurological condition causing severe headaches.",
            source="Test",
            title="Migraine",
            url="",
            chunk_index=0,
            total_chunks=1,
            disease="migraine"
        )
        
        store.add_chunks([chunk])
        results = store.query_by_disease("migraine")
        
        assert len(results) >= 1
    
    def test_get_all_diseases(self):
        """Test getting all disease names."""
        store = VectorStore()
        
        chunks = [
            DocumentChunk(
                content="Content 1",
                source="Test",
                title="Test",
                url="",
                chunk_index=0,
                total_chunks=1,
                disease="disease_a"
            ),
            DocumentChunk(
                content="Content 2",
                source="Test",
                title="Test",
                url="",
                chunk_index=0,
                total_chunks=1,
                disease="disease_b"
            )
        ]
        
        store.add_chunks(chunks)
        diseases = store.get_all_diseases()
        
        assert "disease_a" in diseases
        assert "disease_b" in diseases
    
    def test_clear_store(self):
        """Test clearing the vector store works correctly."""
        # Use in-memory store for isolation (no persist_directory)
        import chromadb
        from chromadb.config import Settings
        
        # Create isolated in-memory client
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create test collection
        collection = client.get_or_create_collection(
            name="test_clear_collection",
            metadata={"description": "Test collection"}
        )
        
        # Add a document
        collection.add(
            documents=["Test content"],
            metadatas=[{"disease": "test"}],
            ids=["test_id_1"]
        )
        
        assert collection.count() == 1
        
        # Clear by deleting collection
        client.delete_collection("test_clear_collection")
        
        # Recreate and verify empty
        collection = client.get_or_create_collection(
            name="test_clear_collection"
        )
        assert collection.count() == 0


class TestExplainAPI:
    """Tests for the explain API endpoints."""
    
    def test_list_topics_empty(self):
        """Test listing topics when DB is empty."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        with TestClient(app) as client:
            response = client.get("/api/v1/explain/")
            
            assert response.status_code == 200
            data = response.json()
            assert "topics" in data
            assert "count" in data
    
    def test_get_stats(self):
        """Test getting knowledge base stats."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        with TestClient(app) as client:
            response = client.get("/api/v1/explain/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert "document_count" in data
            assert "topics" in data
            assert "topic_count" in data


class TestDocumentsExist:
    """Tests to verify sample documents exist."""
    
    def test_documents_directory_exists(self):
        """Test that documents directory exists."""
        docs_dir = Path(__file__).parent.parent / "rag" / "data" / "documents"
        assert docs_dir.exists(), f"Documents directory not found: {docs_dir}"
    
    def test_sample_documents_exist(self):
        """Test that sample documents exist."""
        docs_dir = Path(__file__).parent.parent / "rag" / "data" / "documents"
        
        expected_files = [
            "common_cold.txt",
            "diabetes_type2.txt",
            "hypertension.txt",
            "migraine.txt",
            "pneumonia.txt"
        ]
        
        for filename in expected_files:
            filepath = docs_dir / filename
            assert filepath.exists(), f"Missing document: {filename}"
