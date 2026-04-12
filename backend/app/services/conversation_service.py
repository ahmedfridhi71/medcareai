"""
LLM Conversation Service for symptom extraction.

Uses Mistral AI via LangChain to conduct medical conversations
and extract symptoms from natural language.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI


# System prompt for the medical assistant
MEDICAL_ASSISTANT_PROMPT = """You are MedCareAI, a friendly medical assistant helping patients describe their symptoms.

Your role:
1. Ask 5-8 targeted questions to understand the patient's symptoms
2. Be empathetic and professional
3. Ask about symptom duration, severity, and related symptoms
4. After gathering enough information, summarize the symptoms

Important rules:
- DO NOT diagnose or prescribe medication
- Ask ONE question at a time
- Use simple, clear language
- If the patient seems to have an emergency, advise them to call emergency services

When you have gathered enough symptoms (usually after 5-8 exchanges), end your response with:
[SYMPTOMS_READY]

Then provide a JSON summary in this exact format:
```json
{"symptoms": ["symptom1", "symptom2", "symptom3"]}
```

Start by greeting the patient and asking what brought them here today."""


@dataclass
class ConversationSession:
    """Stores conversation history and state."""
    session_id: str
    messages: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    symptoms_extracted: bool = False
    extracted_symptoms: List[str] = field(default_factory=list)


# In-memory session storage (use Redis in production)
_sessions: Dict[str, ConversationSession] = {}


def get_llm(api_key: str) -> ChatMistralAI:
    """Create Mistral AI chat instance."""
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.7,
        max_tokens=500
    )


def create_session() -> str:
    """Create a new conversation session."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = ConversationSession(session_id=session_id)
    return session_id


def get_session(session_id: str) -> Optional[ConversationSession]:
    """Get an existing session."""
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


async def chat(
    session_id: str,
    user_message: str,
    api_key: str
) -> dict:
    """
    Process a chat message and return AI response.
    
    Args:
        session_id: The conversation session ID
        user_message: The user's message
        api_key: Mistral API key
        
    Returns:
        dict with 'response', 'session_id', 'symptoms_ready'
    """
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    # Build message history for LangChain
    messages = [SystemMessage(content=MEDICAL_ASSISTANT_PROMPT)]
    
    for msg in session.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Add new user message
    messages.append(HumanMessage(content=user_message))
    session.messages.append({"role": "user", "content": user_message})
    
    try:
        # Get LLM response
        llm = get_llm(api_key)
        response = await llm.ainvoke(messages)
        ai_response = response.content
        
        # Store AI response
        session.messages.append({"role": "assistant", "content": ai_response})
        
        # Check if symptoms are ready
        symptoms_ready = "[SYMPTOMS_READY]" in ai_response
        
        if symptoms_ready:
            session.symptoms_extracted = True
            session.extracted_symptoms = extract_symptoms_from_response(ai_response)
        
        return {
            "response": ai_response,
            "session_id": session_id,
            "symptoms_ready": symptoms_ready
        }
        
    except Exception as e:
        # Handle LLM errors gracefully
        error_type = type(e).__name__
        
        if "timeout" in str(e).lower():
            error_msg = "I'm having trouble connecting right now. Please try again in a moment."
        elif "rate" in str(e).lower() or "limit" in str(e).lower():
            error_msg = "I'm receiving too many requests. Please wait a moment and try again."
        elif "invalid" in str(e).lower() or "api" in str(e).lower():
            error_msg = "There's a configuration issue. Please contact support."
        else:
            error_msg = f"I encountered an error ({error_type}). Please try again."
        
        return {
            "response": error_msg,
            "session_id": session_id,
            "symptoms_ready": False,
            "error": str(e)
        }


def extract_symptoms_from_response(response: str) -> List[str]:
    """
    Extract symptoms list from LLM response containing JSON.
    
    Args:
        response: The AI response containing symptom JSON
        
    Returns:
        List of extracted symptom strings
    """
    # Try to find JSON in the response
    json_patterns = [
        r'```json\s*({.*?})\s*```',  # Code block
        r'```\s*({.*?})\s*```',       # Generic code block
        r'({[^{}]*"symptoms"[^{}]*})', # Inline JSON
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                data = json.loads(match.group(1))
                if "symptoms" in data and isinstance(data["symptoms"], list):
                    return data["symptoms"]
            except json.JSONDecodeError:
                continue
    
    return []


def map_symptoms_to_vocabulary(
    extracted_symptoms: List[str],
    vocabulary: List[str],
    threshold: float = 0.6
) -> List[str]:
    """
    Map extracted symptom strings to official vocabulary using fuzzy matching.
    
    Args:
        extracted_symptoms: Raw symptoms from LLM
        vocabulary: List of valid symptom names
        threshold: Minimum similarity threshold (0-1)
        
    Returns:
        List of matched symptoms from vocabulary
    """
    mapped = []
    vocabulary_lower = {v.lower(): v for v in vocabulary}
    vocab_keys = list(vocabulary_lower.keys())
    
    for symptom in extracted_symptoms:
        symptom_lower = symptom.lower().strip()
        
        # Check for exact match first
        if symptom_lower in vocabulary_lower:
            mapped.append(vocabulary_lower[symptom_lower])
            continue
        
        # Fuzzy match
        matches = get_close_matches(
            symptom_lower, 
            vocab_keys, 
            n=1, 
            cutoff=threshold
        )
        
        if matches:
            mapped.append(vocabulary_lower[matches[0]])
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in mapped:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    
    return unique


async def finalize_session(
    session_id: str,
    api_key: str,
    vocabulary: List[str]
) -> dict:
    """
    Finalize a conversation session and extract mapped symptoms.
    
    If symptoms weren't naturally extracted during chat, 
    asks LLM to summarize the conversation.
    
    Args:
        session_id: The conversation session ID
        api_key: Mistral API key
        vocabulary: List of valid symptom names
        
    Returns:
        dict with 'symptoms' (mapped to vocabulary), 'raw_symptoms', 'session_id'
    """
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    # If symptoms already extracted, use them
    if session.symptoms_extracted and session.extracted_symptoms:
        raw_symptoms = session.extracted_symptoms
    else:
        # Ask LLM to extract symptoms from conversation
        raw_symptoms = await extract_symptoms_from_conversation(session, api_key)
        session.extracted_symptoms = raw_symptoms
        session.symptoms_extracted = True
    
    # Map to vocabulary
    mapped_symptoms = map_symptoms_to_vocabulary(raw_symptoms, vocabulary)
    
    return {
        "session_id": session_id,
        "raw_symptoms": raw_symptoms,
        "symptoms": mapped_symptoms,
        "message_count": len(session.messages)
    }


async def extract_symptoms_from_conversation(
    session: ConversationSession,
    api_key: str
) -> List[str]:
    """
    Extract symptoms from conversation history using LLM.
    
    Args:
        session: The conversation session
        api_key: Mistral API key
        
    Returns:
        List of extracted symptoms
    """
    if not session.messages:
        return []
    
    # Build conversation summary for extraction
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in session.messages
    ])
    
    extraction_prompt = f"""Based on the following medical conversation, extract all symptoms mentioned by the patient.

CONVERSATION:
{conversation_text}

Extract symptoms and return ONLY a JSON object in this format:
{{"symptoms": ["symptom1", "symptom2", "symptom3"]}}

Be thorough - include all physical and mental symptoms mentioned."""

    try:
        llm = get_llm(api_key)
        response = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
        return extract_symptoms_from_response(response.content)
    except Exception:
        return []


def get_conversation_history(session_id: str) -> Optional[List[dict]]:
    """Get the full conversation history for a session."""
    session = get_session(session_id)
    if session:
        return session.messages
    return None


def get_active_sessions_count() -> int:
    """Get count of active sessions (for monitoring)."""
    return len(_sessions)
