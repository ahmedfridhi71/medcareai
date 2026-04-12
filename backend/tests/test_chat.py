"""
Chat API Integration Tests.

Tests the LLM conversation endpoints with mocked Mistral API.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.services import conversation_service


class TestConversationService:
    """Tests for the conversation service functions."""
    
    def test_create_session(self):
        """Test creating a new session."""
        session_id = conversation_service.create_session()
        
        assert session_id is not None
        assert len(session_id) == 36  # UUID format
        
        session = conversation_service.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.messages == []
        
        # Cleanup
        conversation_service.delete_session(session_id)
    
    def test_delete_session(self):
        """Test deleting a session."""
        session_id = conversation_service.create_session()
        
        assert conversation_service.delete_session(session_id) is True
        assert conversation_service.get_session(session_id) is None
        assert conversation_service.delete_session(session_id) is False
    
    def test_extract_symptoms_from_response(self):
        """Test extracting symptoms from LLM response."""
        # Test with code block
        response1 = """
        Based on our conversation, you mentioned:
        [SYMPTOMS_READY]
        ```json
        {"symptoms": ["headache", "fever", "fatigue"]}
        ```
        """
        symptoms1 = conversation_service.extract_symptoms_from_response(response1)
        assert symptoms1 == ["headache", "fever", "fatigue"]
        
        # Test with inline JSON
        response2 = 'I see. Your symptoms are: {"symptoms": ["cough", "sore throat"]}'
        symptoms2 = conversation_service.extract_symptoms_from_response(response2)
        assert symptoms2 == ["cough", "sore throat"]
        
        # Test with no JSON
        response3 = "Thank you for sharing. Let me ask more questions."
        symptoms3 = conversation_service.extract_symptoms_from_response(response3)
        assert symptoms3 == []
    
    def test_map_symptoms_to_vocabulary(self):
        """Test mapping symptoms to vocabulary with fuzzy matching."""
        vocabulary = [
            "headache",
            "fever",
            "cough",
            "fatigue",
            "sore_throat",
            "muscle_pain",
            "nausea"
        ]
        
        # Exact matches
        extracted = ["headache", "fever", "cough"]
        mapped = conversation_service.map_symptoms_to_vocabulary(extracted, vocabulary)
        assert mapped == ["headache", "fever", "cough"]
        
        # With fuzzy matching
        extracted2 = ["head ache", "fevers", "throat pain"]
        mapped2 = conversation_service.map_symptoms_to_vocabulary(extracted2, vocabulary, threshold=0.5)
        assert "headache" in mapped2
        
        # Duplicates should be removed
        extracted3 = ["headache", "Headache", "HEADACHE"]
        mapped3 = conversation_service.map_symptoms_to_vocabulary(extracted3, vocabulary)
        assert mapped3 == ["headache"]


class TestChatAPIEndpoints:
    """Tests for the chat API endpoints."""
    
    def test_start_chat(self):
        """Test starting a new chat session."""
        with TestClient(app) as client:
            response = client.post("/api/v1/chat/start")
            
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert "message" in data
            assert len(data["session_id"]) == 36
            
            # Cleanup
            conversation_service.delete_session(data["session_id"])
    
    @patch("app.services.conversation_service.get_llm")
    def test_send_message(self, mock_get_llm):
        """Test sending a message with mocked LLM."""
        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I understand you have a headache. How long have you been experiencing this?"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm
        
        with TestClient(app) as client:
            # Start session
            start_response = client.post("/api/v1/chat/start")
            session_id = start_response.json()["session_id"]
            
            # Send message
            response = client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": session_id,
                    "message": "I have a terrible headache"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert "headache" in data["response"].lower()
            assert data["symptoms_ready"] is False
            
            # Cleanup
            conversation_service.delete_session(session_id)
    
    def test_send_message_invalid_session(self):
        """Test sending message to non-existent session."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": "non-existent-session-id",
                    "message": "Hello"
                }
            )
            
            assert response.status_code == 404
    
    @patch("app.services.conversation_service.extract_symptoms_from_conversation", new_callable=AsyncMock)
    def test_finalize_chat(self, mock_extract):
        """Test finalizing chat and extracting symptoms."""
        # Setup mock for extraction
        mock_extract.return_value = ["headache", "fever", "fatigue"]
        
        with TestClient(app) as client:
            # Start session and add some messages
            start_response = client.post("/api/v1/chat/start")
            session_id = start_response.json()["session_id"]
            
            # Add a message to the session directly
            session = conversation_service.get_session(session_id)
            session.messages.append({"role": "user", "content": "I have headache and fever"})
            session.messages.append({"role": "assistant", "content": "I see. Any fatigue?"})
            session.messages.append({"role": "user", "content": "Yes, I'm very tired"})
            
            # Finalize
            response = client.post(
                "/api/v1/chat/finalize",
                json={"session_id": session_id}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
            assert len(data["raw_symptoms"]) > 0
            assert data["message_count"] == 3
            
            # Cleanup
            conversation_service.delete_session(session_id)
    
    def test_get_history(self):
        """Test getting conversation history."""
        with TestClient(app) as client:
            # Start session
            start_response = client.post("/api/v1/chat/start")
            session_id = start_response.json()["session_id"]
            
            # Add some messages directly
            session = conversation_service.get_session(session_id)
            session.messages.append({"role": "user", "content": "Hello"})
            session.messages.append({"role": "assistant", "content": "Hi there!"})
            
            # Get history
            response = client.get(f"/api/v1/chat/history/{session_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["messages"]) == 2
            
            # Cleanup
            conversation_service.delete_session(session_id)
    
    def test_get_history_invalid_session(self):
        """Test getting history for non-existent session."""
        with TestClient(app) as client:
            response = client.get("/api/v1/chat/history/non-existent")
            assert response.status_code == 404
    
    def test_delete_session(self):
        """Test deleting a session via API."""
        with TestClient(app) as client:
            # Start session
            start_response = client.post("/api/v1/chat/start")
            session_id = start_response.json()["session_id"]
            
            # Delete it
            response = client.delete(f"/api/v1/chat/{session_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["deleted"] is True
            
            # Try to delete again
            response2 = client.delete(f"/api/v1/chat/{session_id}")
            assert response2.json()["deleted"] is False
    
    def test_sessions_count(self):
        """Test getting active sessions count."""
        with TestClient(app) as client:
            initial_response = client.get("/api/v1/chat/sessions/count")
            initial_count = initial_response.json()["active_sessions"]
            
            # Create sessions
            session_ids = []
            for _ in range(3):
                resp = client.post("/api/v1/chat/start")
                session_ids.append(resp.json()["session_id"])
            
            # Check count increased
            response = client.get("/api/v1/chat/sessions/count")
            assert response.json()["active_sessions"] == initial_count + 3
            
            # Cleanup
            for sid in session_ids:
                conversation_service.delete_session(sid)


class TestFullConversationFlow:
    """Integration test for a complete conversation flow."""
    
    @patch("app.services.conversation_service.extract_symptoms_from_conversation", new_callable=AsyncMock)
    @patch("app.services.conversation_service.get_llm")
    def test_full_flow_chat_to_prediction(self, mock_get_llm, mock_extract):
        """Test complete flow: start → chat → finalize → predict."""
        # Setup mock for extraction
        mock_extract.return_value = ["headache", "fever", "fatigue"]
        
        # Setup mock with conversation simulation
        responses = [
            "I understand you have a headache. How long have you experienced this?",
            "I see, 3 days. Have you noticed any fever or chills?",
            "Thank you for that information. Any other symptoms like fatigue or nausea?",
            """Based on our conversation, you mentioned:
            [SYMPTOMS_READY]
            ```json
            {"symptoms": ["headache", "fever", "fatigue"]}
            ```
            Please consult a healthcare provider for proper diagnosis."""
        ]
        response_iter = iter(responses)
        
        def mock_response_generator(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.content = next(response_iter)
            return mock_resp
        
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=mock_response_generator)
        mock_get_llm.return_value = mock_llm
        
        with TestClient(app) as client:
            # 1. Start chat
            start_resp = client.post("/api/v1/chat/start")
            assert start_resp.status_code == 200
            session_id = start_resp.json()["session_id"]
            
            # 2. Chat conversation
            messages = [
                "I've been having terrible headaches",
                "About 3 days now",
                "Yes, I've had some fever and feeling very tired"
            ]
            
            last_response = None
            for msg in messages:
                resp = client.post(
                    "/api/v1/chat/message",
                    json={"session_id": session_id, "message": msg}
                )
                assert resp.status_code == 200
                last_response = resp.json()
            
            # Last response should have symptoms ready
            # Note: Due to our mock, we control when this happens
            
            # 3. Finalize and extract symptoms
            finalize_resp = client.post(
                "/api/v1/chat/finalize",
                json={"session_id": session_id}
            )
            assert finalize_resp.status_code == 200
            finalize_data = finalize_resp.json()
            assert len(finalize_data["raw_symptoms"]) > 0
            
            # 4. If we have mapped symptoms, we could call predict
            # (Not testing actual prediction here as it depends on model)
            
            # Cleanup
            conversation_service.delete_session(session_id)
