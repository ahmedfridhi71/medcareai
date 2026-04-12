"""
Document Chunker Module.

Splits documents into smaller chunks for embedding and retrieval.
Uses 512-token chunks with 50-token overlap to preserve context.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import PyPDF2


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    source: str
    title: str
    url: str
    chunk_index: int
    total_chunks: int
    disease: Optional[str] = None


def extract_metadata_from_text(content: str) -> dict:
    """
    Extract metadata (title, source, URL) from document header.
    
    Expects format:
    TITLE: ...
    SOURCE: ...
    URL: ...
    """
    metadata = {
        "title": "Unknown",
        "source": "Unknown",
        "url": ""
    }
    
    lines = content.split("\n")[:10]  # Check first 10 lines
    
    for line in lines:
        line = line.strip()
        if line.startswith("TITLE:"):
            metadata["title"] = line[6:].strip()
        elif line.startswith("SOURCE:"):
            metadata["source"] = line[7:].strip()
        elif line.startswith("URL:"):
            metadata["url"] = line[4:].strip()
    
    return metadata


def extract_disease_from_title(title: str) -> str:
    """
    Extract disease name from document title.
    
    Examples:
        "Common Cold (Acute Viral Rhinopharyngitis)" -> "common cold"
        "Type 2 Diabetes Mellitus" -> "type 2 diabetes mellitus"
    """
    # Remove parenthetical content
    disease = re.sub(r'\([^)]*\)', '', title)
    # Clean up and lowercase
    disease = disease.strip().lower()
    return disease


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (roughly 4 characters per token).
    
    This is a simple heuristic; actual tokenizers may differ.
    """
    return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving structure."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Target size in tokens (approximate)
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens_approx(sentence)
        
        # If single sentence exceeds chunk size, split it
        if sentence_tokens > chunk_size:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = count_tokens_approx(word)
                if temp_tokens + word_tokens > chunk_size:
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
            
            if temp_chunk:
                current_chunk = temp_chunk
                current_tokens = temp_tokens
            continue
        
        # Check if adding this sentence exceeds chunk size
        if current_tokens + sentence_tokens > chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                # Take last few sentences as overlap
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if count_tokens_approx(overlap_text + s) < overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = " ".join(overlap_sentences)
                    else:
                        break
                current_chunk = overlap_sentences + [sentence]
                current_tokens = count_tokens_approx(" ".join(current_chunk))
            else:
                current_chunk = [sentence]
                current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def load_text_file(file_path: Path) -> str:
    """Load content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf_file(file_path: Path) -> str:
    """Load content from a PDF file."""
    text = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n\n".join(text)


def load_document(file_path: Path) -> str:
    """Load document content based on file type."""
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return load_pdf_file(file_path)
    elif suffix in ['.txt', '.md', '.text']:
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def process_document(
    file_path: Path,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[DocumentChunk]:
    """
    Process a document into chunks with metadata.
    
    Args:
        file_path: Path to the document
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        List of DocumentChunk objects
    """
    # Load document
    content = load_document(file_path)
    
    # Extract metadata
    metadata = extract_metadata_from_text(content)
    disease = extract_disease_from_title(metadata["title"])
    
    # Remove metadata lines from content for chunking
    content_lines = content.split("\n")
    content_start = 0
    for i, line in enumerate(content_lines[:10]):
        if line.strip().startswith(("TITLE:", "SOURCE:", "URL:")):
            content_start = i + 1
    
    clean_content = "\n".join(content_lines[content_start:]).strip()
    
    # Chunk the content
    chunks = chunk_text(clean_content, chunk_size, overlap)
    
    # Create DocumentChunk objects
    document_chunks = []
    for i, chunk_content in enumerate(chunks):
        document_chunks.append(DocumentChunk(
            content=chunk_content,
            source=metadata["source"],
            title=metadata["title"],
            url=metadata["url"],
            chunk_index=i,
            total_chunks=len(chunks),
            disease=disease
        ))
    
    return document_chunks


def process_documents_directory(
    directory: Path,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[DocumentChunk]:
    """
    Process all documents in a directory.
    
    Args:
        directory: Path to directory containing documents
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks
        
    Returns:
        List of all DocumentChunk objects
    """
    all_chunks = []
    
    # Supported extensions
    extensions = ['.txt', '.pdf', '.md', '.text']
    
    for ext in extensions:
        for file_path in directory.glob(f'*{ext}'):
            try:
                chunks = process_document(file_path, chunk_size, overlap)
                all_chunks.extend(chunks)
                print(f"Processed {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
    
    return all_chunks
