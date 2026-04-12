"""
Explanation API Endpoints.

Provides RAG-powered medical explanations using scientific sources.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from rag.rag_service import (
    generate_explanation,
    get_available_topics,
    quick_answer,
)
from rag.vector_store import get_vector_store

router = APIRouter(tags=["Explain"])


# Request/Response Models
class ExplanationResponse(BaseModel):
    """Response with disease explanation and sources."""
    disease: str = Field(..., description="Disease name")
    explanation: str = Field(..., description="Patient-friendly explanation")
    sources: List[dict] = Field(default_factory=list, description="Source citations")
    chunks_used: int = Field(..., description="Number of knowledge chunks used")


class QuestionRequest(BaseModel):
    """Request for a quick medical question."""
    question: str = Field(..., min_length=5, max_length=500, description="Medical question")


class QuestionResponse(BaseModel):
    """Response to a medical question."""
    answer: str = Field(..., description="Answer based on sources")
    sources: List[dict] = Field(default_factory=list, description="Source citations")
    chunks_used: int = Field(..., description="Number of chunks used")


class TopicsResponse(BaseModel):
    """List of available topics in knowledge base."""
    topics: List[str] = Field(default_factory=list, description="Available disease/topic names")
    count: int = Field(..., description="Total number of topics")


class KnowledgeBaseStats(BaseModel):
    """Statistics about the knowledge base."""
    document_count: int = Field(..., description="Total document chunks")
    topics: List[str] = Field(default_factory=list, description="Available topics")
    topic_count: int = Field(..., description="Number of unique topics")


# Dependency to get vector store
def get_store():
    """Get the vector store instance."""
    persist_dir = settings.chroma_persist_directory
    return get_vector_store(persist_dir)


# IMPORTANT: Static routes MUST come before dynamic routes like /{disease_name}

@router.get(
    "/",
    response_model=TopicsResponse,
    summary="List available topics",
    description="Returns all topics available in the knowledge base."
)
async def list_topics() -> TopicsResponse:
    """
    Get list of all diseases/topics in the knowledge base.
    
    Use these names when requesting explanations.
    """
    try:
        store = get_store()
        topics = get_available_topics(vector_store=store)
        
        return TopicsResponse(
            topics=topics,
            count=len(topics)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing topics: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=KnowledgeBaseStats,
    summary="Get knowledge base stats",
    description="Returns statistics about the knowledge base."
)
async def get_stats() -> KnowledgeBaseStats:
    """
    Get statistics about the medical knowledge base.
    """
    try:
        store = get_store()
        topics = store.get_all_diseases()
        doc_count = store.get_document_count()
        
        return KnowledgeBaseStats(
            document_count=doc_count,
            topics=topics,
            topic_count=len(topics)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


@router.post(
    "/question",
    response_model=QuestionResponse,
    summary="Ask a medical question",
    description="Answer a medical question using the knowledge base."
)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Ask a medical question and get a source-based answer.
    
    The answer is generated using RAG from the medical knowledge base.
    """
    if not settings.mistral_api_key:
        raise HTTPException(
            status_code=500,
            detail="Mistral API key not configured"
        )
    
    try:
        store = get_store()
        
        result = await quick_answer(
            question=request.question,
            api_key=settings.mistral_api_key,
            vector_store=store
        )
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )


# Dynamic route MUST be last - it catches all paths
@router.get(
    "/{disease_name}",
    response_model=ExplanationResponse,
    summary="Get disease explanation",
    description="Returns an evidence-based explanation of a disease using RAG."
)
async def explain_disease(
    disease_name: str,
    n_chunks: int = 5
) -> ExplanationResponse:
    """
    Get a patient-friendly explanation of a disease.
    
    Uses RAG to retrieve relevant medical literature and generate
    an accurate, source-cited explanation.
    """
    if not settings.mistral_api_key:
        raise HTTPException(
            status_code=500,
            detail="Mistral API key not configured"
        )
    
    try:
        store = get_store()
        
        result = await generate_explanation(
            disease_name=disease_name,
            api_key=settings.mistral_api_key,
            vector_store=store,
            n_chunks=n_chunks
        )
        
        return ExplanationResponse(
            disease=result.disease,
            explanation=result.explanation,
            sources=[
                {
                    "title": s.title,
                    "source": s.source,
                    "url": s.url,
                    "relevance": s.relevance
                }
                for s in result.sources
            ],
            chunks_used=result.chunks_used
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation: {str(e)}"
        )
