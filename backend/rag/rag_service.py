"""
RAG Service Module.

Combines vector retrieval with LLM generation for
evidence-based medical explanations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI

from rag.vector_store import VectorStore, get_vector_store


@dataclass
class Source:
    """A source citation for an explanation."""
    title: str
    source: str
    url: str
    relevance: float


@dataclass
class ExplanationResult:
    """Result of a RAG explanation query."""
    disease: str
    explanation: str
    sources: List[Source]
    chunks_used: int


# System prompt for medical explanation
EXPLANATION_PROMPT = """You are MedCareAI, a medical information assistant providing evidence-based health information.

Your task is to explain a medical condition to a patient using ONLY the provided medical sources.

IMPORTANT RULES:
1. Use ONLY information from the provided sources - do not add external knowledge
2. Explain in clear, simple language suitable for patients
3. Be accurate and cite which source each fact comes from
4. Include sections on: Overview, Symptoms, Causes, Treatment, When to See a Doctor
5. Do NOT provide specific medical advice or diagnoses
6. Always recommend consulting a healthcare provider

Format your response in clear sections with headers."""


def format_chunks_for_prompt(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks for inclusion in LLM prompt.
    
    Args:
        chunks: List of chunk dictionaries from vector store
        
    Returns:
        Formatted string of all chunks with source info
    """
    formatted = []
    
    for i, chunk in enumerate(chunks, 1):
        source_info = chunk.get("metadata", {})
        formatted.append(f"""
--- SOURCE {i}: {source_info.get('title', 'Unknown')} ---
From: {source_info.get('source', 'Unknown')}

{chunk['content']}
""")
    
    return "\n".join(formatted)


def extract_sources(chunks: List[Dict]) -> List[Source]:
    """
    Extract unique sources from chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of unique Source objects
    """
    seen_titles = set()
    sources = []
    
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Unknown")
        
        if title not in seen_titles:
            seen_titles.add(title)
            sources.append(Source(
                title=title,
                source=meta.get("source", "Unknown"),
                url=meta.get("url", ""),
                relevance=chunk.get("relevance_score", 0.0)
            ))
    
    return sources


async def generate_explanation(
    disease_name: str,
    api_key: str,
    vector_store: Optional[VectorStore] = None,
    persist_directory: Optional[str] = None,
    n_chunks: int = 5
) -> ExplanationResult:
    """
    Generate an evidence-based explanation for a disease.
    
    Args:
        disease_name: Name of the disease to explain
        api_key: Mistral API key
        vector_store: Optional pre-configured vector store
        persist_directory: Directory for vector store (if not provided)
        n_chunks: Number of chunks to retrieve
        
    Returns:
        ExplanationResult with explanation and sources
    """
    # Get vector store
    store = vector_store or get_vector_store(persist_directory)
    
    # Retrieve relevant chunks
    chunks = store.query_by_disease(disease_name, n_results=n_chunks)
    
    if not chunks:
        return ExplanationResult(
            disease=disease_name,
            explanation=f"I don't have enough information about {disease_name} in my knowledge base. "
                       f"Please consult a healthcare provider for accurate information.",
            sources=[],
            chunks_used=0
        )
    
    # Format chunks for prompt
    context = format_chunks_for_prompt(chunks)
    
    # Create LLM prompt
    user_prompt = f"""Using the following medical sources, explain "{disease_name}" to a patient:

{context}

Provide a comprehensive but easy-to-understand explanation covering:
1. What is {disease_name}?
2. Common symptoms
3. Causes and risk factors
4. Treatment options
5. When to seek medical care

Remember to use simple language and cite sources."""

    # Generate explanation with LLM
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.3,  # Lower temperature for factual content
        max_tokens=1500
    )
    
    messages = [
        SystemMessage(content=EXPLANATION_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    
    # Extract sources
    sources = extract_sources(chunks)
    
    return ExplanationResult(
        disease=disease_name,
        explanation=response.content,
        sources=sources,
        chunks_used=len(chunks)
    )


async def quick_answer(
    question: str,
    api_key: str,
    vector_store: Optional[VectorStore] = None,
    persist_directory: Optional[str] = None,
    n_chunks: int = 3
) -> Dict:
    """
    Answer a quick medical question using RAG.
    
    Args:
        question: User's question
        api_key: Mistral API key
        vector_store: Optional pre-configured vector store
        persist_directory: Directory for vector store
        n_chunks: Number of chunks to retrieve
        
    Returns:
        Dict with answer and sources
    """
    # Get vector store
    store = vector_store or get_vector_store(persist_directory)
    
    # Retrieve relevant chunks
    chunks = store.query(question, n_results=n_chunks)
    
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in my knowledge base. "
                     "Please consult a healthcare provider.",
            "sources": [],
            "chunks_used": 0
        }
    
    context = format_chunks_for_prompt(chunks)
    
    user_prompt = f"""Based on the following medical sources, answer this question:

Question: {question}

Sources:
{context}

Provide a clear, accurate answer based ONLY on the sources. If the sources don't contain 
enough information, say so. Always recommend consulting a healthcare provider for personalized advice."""

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.3,
        max_tokens=500
    )
    
    messages = [
        SystemMessage(content="You are a medical information assistant. Provide accurate, "
                            "source-based answers. Always recommend professional medical advice."),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    sources = extract_sources(chunks)
    
    return {
        "answer": response.content,
        "sources": [
            {"title": s.title, "source": s.source, "url": s.url}
            for s in sources
        ],
        "chunks_used": len(chunks)
    }


def get_available_topics(
    vector_store: Optional[VectorStore] = None,
    persist_directory: Optional[str] = None
) -> List[str]:
    """
    Get list of available topics in the knowledge base.
    
    Args:
        vector_store: Optional pre-configured vector store
        persist_directory: Directory for vector store
        
    Returns:
        List of disease/topic names
    """
    store = vector_store or get_vector_store(persist_directory)
    return store.get_all_diseases()
