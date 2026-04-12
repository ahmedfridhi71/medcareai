#!/usr/bin/env python
"""
Rebuild Vector Database Script.

Processes all documents in the data directory and rebuilds
the ChromaDB vector store.

Usage:
    python scripts/rebuild_vectordb.py [--clear]
    
Options:
    --clear     Clear existing database before rebuilding
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chunker import process_documents_directory
from rag.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the MedCareAI vector database"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing database before rebuilding"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="rag/data/documents",
        help="Directory containing documents (default: rag/data/documents)"
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="rag/chroma_db",
        help="Directory for ChromaDB storage (default: rag/chroma_db)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / args.docs_dir
    db_dir = base_dir / args.db_dir
    
    print("=" * 60)
    print("MedCareAI Vector Database Rebuild")
    print("=" * 60)
    print(f"\nDocuments directory: {docs_dir}")
    print(f"Database directory: {db_dir}")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Overlap: {args.overlap} tokens")
    
    # Check if documents directory exists
    if not docs_dir.exists():
        print(f"\nError: Documents directory not found: {docs_dir}")
        sys.exit(1)
    
    # Initialize vector store
    print("\nInitializing vector store...")
    store = VectorStore(persist_directory=str(db_dir))
    
    # Clear if requested
    if args.clear:
        print("Clearing existing database...")
        store.clear()
        print("Database cleared.")
    
    # Get current stats
    initial_count = store.get_document_count()
    print(f"Current document count: {initial_count}")
    
    # Process documents
    print(f"\nProcessing documents from {docs_dir}...")
    chunks = process_documents_directory(
        docs_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if not chunks:
        print("No documents found to process.")
        sys.exit(0)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Add chunks to vector store
    print("\nAdding chunks to vector store...")
    added = store.add_chunks(chunks)
    
    # Final stats
    final_count = store.get_document_count()
    topics = store.get_all_diseases()
    
    print("\n" + "=" * 60)
    print("Rebuild Complete!")
    print("=" * 60)
    print(f"Documents added: {added}")
    print(f"Total documents: {final_count}")
    print(f"Topics covered: {len(topics)}")
    print(f"\nTopics: {', '.join(topics)}")
    print("\nVector database is ready for use.")


if __name__ == "__main__":
    main()
