"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request schema for creating a new session."""
    user_id: str = Field(..., description="User identifier")
    platform: Optional[str] = Field("linkedin", description="Default platform for this session")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "platform": "linkedin"
            }
        }


class CreateSessionResponse(BaseModel):
    """Response schema for session creation."""
    session_id: str = Field(..., description="Generated session identifier")
    user_id: str = Field(..., description="User identifier")
    platform: str = Field(..., description="Default platform")
    message: str = Field(..., description="Success message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc123",
                "user_id": "user-123",
                "platform": "linkedin",
                "message": "Session created successfully"
            }
        }


class GeneratePostRequest(BaseModel):
    """Request schema for post generation."""
    session_id: str = Field(..., description="Session identifier")
    topic: str = Field(..., min_length=3, max_length=500, description="Topic to write about")
    platform: Optional[str] = Field(None, description="Target platform (linkedin, twitter, etc.)")
    tone: Optional[str] = Field(None, description="Writing tone (professional, casual, insightful, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-abc123",
                "topic": "The impact of transformer models on natural language processing",
                "platform": "linkedin",
                "tone": "insightful"
            }
        }


class GeneratePostResponse(BaseModel):
    """Response schema for post generation."""
    post_markdown: str = Field(..., description="Generated post in markdown format")
    scores: Dict[str, float] = Field(..., description="Quality scores (faithfulness, answer_relevancy)")
    session_id: str = Field(..., description="Session identifier")
    refinement_count: int = Field(..., description="Number of refinement iterations performed")
    trace_url: Optional[str] = Field(None, description="Langfuse trace URL for observability")
    
    class Config:
        json_schema_extra = {
            "example": {
                "post_markdown": "# The Transformer Revolution\n\nTransformers have revolutionized NLP...",
                "scores": {
                    "faithfulness": 0.92,
                    "answer_relevancy": 0.87
                },
                "session_id": "session-abc123",
                "refinement_count": 0,
                "trace_url": "https://cloud.langfuse.com/trace/trace-xyz789"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")
    version: str = Field("0.1.0", description="API version")
    components: Dict[str, str] = Field(..., description="Component statuses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "All systems operational",
                "version": "0.1.0",
                "components": {
                    "retriever": "ok",
                    "ltm": "ok",
                    "checkpointer": "ok",
                    "tracer": "ok"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Any] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "detail": {"field": "topic", "issue": "too short"}
            }
        }
