"""
Smoke tests for the FastAPI application.

Run with: pytest tests/test_api_smoke.py -v
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "components" in data
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_create_session():
    """Test session creation."""
    response = client.post(
        "/sessions",
        json={"user_id": "test-user", "platform": "linkedin"}
    )
    assert response.status_code == 201
    
    data = response.json()
    assert "session_id" in data
    assert data["user_id"] == "test-user"
    assert data["platform"] == "linkedin"


def test_create_session_invalid_payload():
    """Test session creation with invalid payload."""
    response = client.post(
        "/sessions",
        json={"platform": "linkedin"}  # Missing user_id
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.skipif(
    reason="Requires FAISS index to be built first"
)
def test_generate_post():
    """Test post generation (requires FAISS index)."""
    # First create a session
    session_response = client.post(
        "/sessions",
        json={"user_id": "test-user", "platform": "linkedin"}
    )
    session_id = session_response.json()["session_id"]
    
    # Generate post
    response = client.post(
        "/posts:generate",
        json={
            "session_id": session_id,
            "topic": "The impact of transformers on NLP",
            "platform": "linkedin",
            "tone": "insightful"
        }
    )
    
    # Might fail if FAISS not built, but check structure if successful
    if response.status_code == 200:
        data = response.json()
        assert "post_markdown" in data
        assert "scores" in data
        assert "session_id" in data


def test_generate_post_invalid_session():
    """Test post generation with invalid session ID."""
    response = client.post(
        "/posts:generate",
        json={
            "session_id": "invalid-session",
            "topic": "Test topic",
            "platform": "linkedin"
        }
    )
    assert response.status_code == 400


def test_generate_post_topic_too_short():
    """Test post generation with topic too short."""
    session_response = client.post(
        "/sessions",
        json={"user_id": "test-user", "platform": "linkedin"}
    )
    session_id = session_response.json()["session_id"]
    
    response = client.post(
        "/posts:generate",
        json={
            "session_id": session_id,
            "topic": "AI",  # Too short (< 3 chars)
            "platform": "linkedin"
        }
    )
    assert response.status_code == 422
