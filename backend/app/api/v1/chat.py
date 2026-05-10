"""
Chat API Endpoints.

Handles LLM-powered medical conversations for symptom extraction.
"""

from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from app.config import settings
from app.schemas.chat import (
    ChatDeleteResponse,
    ChatFinalizeRequest,
    ChatFinalizeResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatStartResponse,
)
from app.services import conversation_service
from app.services.ml_service import get_symptoms_vocabulary

load_dotenv()

# ✅ FIX: REMOVE prefix="/chat"
router = APIRouter(tags=["Chat"])


@router.post("/start", response_model=ChatStartResponse)
async def start_chat():
    session_id = conversation_service.create_session()

    return ChatStartResponse(
        session_id=session_id,
        message="Hello! I'm MedCareAI, your medical assistant. What symptoms are you experiencing today?"
    )


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest):
    if not settings.mistral_api_key:
        raise HTTPException(
            status_code=500,
            detail="Mistral API key not configured"
        )
    
    try:
        result = await conversation_service.chat(
            session_id=request.session_id,
            user_message=request.message,
            api_key=settings.mistral_api_key
        )
        
        return ChatMessageResponse(
            session_id=result["session_id"],
            response=result["response"],
            symptoms_ready=result.get("symptoms_ready", False),
            error=result.get("error")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/finalize", response_model=ChatFinalizeResponse)
async def finalize_chat(request: ChatFinalizeRequest):
    if not settings.mistral_api_key:
        raise HTTPException(
            status_code=500,
            detail="Mistral API key not configured"
        )
    
    try:
        vocabulary = get_symptoms_vocabulary()
        
        result = await conversation_service.finalize_session(
            session_id=request.session_id,
            api_key=settings.mistral_api_key,
            vocabulary=vocabulary
        )
        
        return ChatFinalizeResponse(
            session_id=result["session_id"],
            raw_symptoms=result["raw_symptoms"],
            symptoms=result["symptoms"],
            message_count=result["message_count"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finalize error: {str(e)}")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    messages = conversation_service.get_conversation_history(session_id)
    
    if messages is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages
    )


@router.delete("/{session_id}", response_model=ChatDeleteResponse)
async def delete_chat_session(session_id: str):
    deleted = conversation_service.delete_session(session_id)
    
    return ChatDeleteResponse(
        session_id=session_id,
        deleted=deleted
    )


@router.get("/sessions/count")
async def get_sessions_count():
    return {"active_sessions": conversation_service.get_active_sessions_count()}