"""
Test configuration and fixtures for MedCareAI backend tests.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def anyio_backend():
    """Use asyncio as the async backend."""
    return "asyncio"


@pytest.fixture
async def client():
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac
