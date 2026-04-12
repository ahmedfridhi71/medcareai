"""
Document Embedder Module.

Uses sentence-transformers to create embeddings for document chunks.
Supports BioBERT and other medical-domain embedding models.
"""

from typing import List, Optional

from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """
    Creates embeddings for document chunks using sentence-transformers.
    
    Default model is optimized for medical/scientific text.
    """
    
    # Default model - good balance of speed and quality for medical text
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Alternative medical-specific model (larger, more accurate)
    MEDICAL_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-sst2"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name. Defaults to MiniLM for speed.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name, device=device)
        self._embedding_dimension = None
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dimension is None:
            # Get dimension by encoding a test string
            test_embedding = self.model.encode("test")
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings (each is a list of floats)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Create embedding for a search query.
        
        This is an alias for embed_text, but may be overridden
        for models that use different encoding for queries vs documents.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed_text(query)


def get_embedder(
    use_medical_model: bool = False,
    model_name: Optional[str] = None
) -> DocumentEmbedder:
    """
    Factory function to create an embedder.
    
    Args:
        use_medical_model: If True, use BioBERT medical model
        model_name: Custom model name (overrides use_medical_model)
        
    Returns:
        Configured DocumentEmbedder instance
    """
    if model_name:
        return DocumentEmbedder(model_name=model_name)
    
    if use_medical_model:
        return DocumentEmbedder(model_name=DocumentEmbedder.MEDICAL_MODEL)
    
    return DocumentEmbedder()
