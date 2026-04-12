"""
Chat Pydantic Schemas.

Request/response models for the LLM conversation endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatStartResponse(BaseModel):
    """Response when starting a new chat session."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="Welcome message from assistant")


class ChatMessageRequest(BaseModel):
    """Request to send a message in a chat session."""
    session_id: str = Field(..., description="The chat session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")


class ChatMessageResponse(BaseModel):
    """Response from the chat assistant."""
    session_id: str = Field(..., description="The chat session ID")
    response: str = Field(..., description="Assistant response")
    symptoms_ready: bool = Field(
        default=False,
        description="True if assistant has gathered enough symptoms"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class ChatFinalizeRequest(BaseModel):
    """Request to finalize a chat session and extract symptoms."""
    session_id: str = Field(..., description="The chat session ID")


class ChatFinalizeResponse(BaseModel):
    """Response with extracted and mapped symptoms."""
    session_id: str = Field(..., description="The chat session ID")
    raw_symptoms: List[str] = Field(
        default_factory=list,
        description="Symptoms as extracted from conversation"
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Symptoms mapped to official vocabulary"
    )
    message_count: int = Field(..., description="Number of messages in conversation")


class ChatHistoryResponse(BaseModel):
    """Response with conversation history."""
    session_id: str = Field(..., description="The chat session ID")
    messages: List[dict] = Field(default_factory=list, description="Conversation messages")


class ChatDeleteResponse(BaseModel):
    """Response when deleting a chat session."""
    session_id: str = Field(..., description="The deleted session ID")
    deleted: bool = Field(..., description="True if session was found and deleted")
