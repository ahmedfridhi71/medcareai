"""
Tests for the main FastAPI application endpoints.
"""

import pytest


@pytest.mark.anyio
async def test_health_check(client):
    """Test that health check endpoint returns healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app" in data
    assert "environment" in data


@pytest.mark.anyio
async def test_root_endpoint(client):
    """Test that root endpoint returns welcome message."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "MedCareAI" in data["message"]
