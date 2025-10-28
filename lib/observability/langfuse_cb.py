"""
Langfuse observability integration for LangChain and LangGraph.

This module provides helpers for integrating Langfuse tracing into the
agentic workflow, including:
- Callback handler initialization
- Custom score attachment
- Trace URL generation

Langfuse provides full observability for LLM applications with:
- Distributed tracing across multi-agent workflows
- Custom metrics and scores
- Performance monitoring
- Cost tracking
"""

import os
from typing import Optional, Dict, Any, TYPE_CHECKING
import warnings

try:
    from langfuse.callback import CallbackHandler
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None  # type: ignore
    Langfuse = None  # type: ignore
    warnings.warn("langfuse package not installed. Tracing will be disabled.")


class LangfuseTracer:
    """
    Langfuse tracing helper for the agentic post generator.
    
    Manages Langfuse initialization and provides utilities for:
    - Creating callback handlers for LangChain/LangGraph
    - Attaching custom scores (faithfulness, answer_relevancy)
    - Generating trace URLs
    
    Attributes:
        enabled (bool): Whether Langfuse is configured and available
        client (Langfuse): Langfuse client instance
        public_key (str): Langfuse public key
        secret_key (str): Langfuse secret key
        host (str): Langfuse host URL
    """
    
    def __init__(self):
        """
        Initialize the Langfuse tracer.
        
        Attempts to configure Langfuse from environment variables:
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_HOST (optional, defaults to cloud.langfuse.com)
        
        If credentials are not configured, tracing will be disabled.
        """
        self.enabled = False
        self.client = None
        
        if not LANGFUSE_AVAILABLE:
            print("Langfuse package not available. Tracing disabled.")
            return
        
        # Load credentials
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if not self.public_key or not self.secret_key:
            print("Langfuse credentials not configured. Tracing disabled.")
            print("   Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable.")
            return
        
        try:
            # Initialize Langfuse client
            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            self.enabled = True
            print(f"Langfuse tracing enabled: {self.host}")
        except Exception as e:
            print(f"Failed to initialize Langfuse: {e}")
            print("   Tracing will be disabled.")
    
    def get_callback_handler(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CallbackHandler]:
        """
        Create a Langfuse callback handler for LangChain/LangGraph.
        
        This callback automatically traces:
        - LLM calls (prompts, completions, token counts)
        - Chain executions
        - Agent steps
        - Tool usage
        
        Args:
            session_id (Optional[str]): Session/thread identifier
            user_id (Optional[str]): User identifier
            metadata (Optional[Dict]): Additional metadata to attach to trace
            
        Returns:
            Optional[CallbackHandler]: Langfuse callback or None if disabled
            
        Usage:
            tracer = LangfuseTracer()
            handler = tracer.get_callback_handler(
                session_id="thread-123",
                user_id="user-456",
                metadata={"platform": "linkedin", "topic": "AI"}
            )
            
            # Use with LangChain
            llm.invoke("prompt", config={"callbacks": [handler]})
            
            # Use with LangGraph
            graph.invoke(state, config={"callbacks": [handler]})
        """
        if not self.enabled:
            return None
        
        try:
            # Build trace metadata
            trace_metadata = metadata or {}
            if session_id:
                trace_metadata["session_id"] = session_id
            if user_id:
                trace_metadata["user_id"] = user_id
            
            # Create callback handler
            handler = CallbackHandler(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
                session_id=session_id,
                user_id=user_id,
                metadata=trace_metadata,
            )
            
            return handler
            
        except Exception as e:
            print(f"Error creating Langfuse callback: {e}")
            return None
    
    def add_scores(
        self,
        trace_id: str,
        scores: Dict[str, float],
        comment: Optional[str] = None,
    ) -> bool:
        """
        Add custom scores to a Langfuse trace.
        
        Used to attach evaluation metrics (faithfulness, answer_relevancy, etc.)
        to a completed trace for analysis.
        
        Args:
            trace_id (str): Langfuse trace ID
            scores (Dict[str, float]): Score name -> value mapping
            comment (Optional[str]): Optional comment about the scores
            
        Returns:
            bool: True if scores were added successfully
            
        Example:
            tracer.add_scores(
                trace_id="trace-abc123",
                scores={
                    "faithfulness": 0.92,
                    "answer_relevancy": 0.87
                },
                comment="DeepEval metrics from fact-checker"
            )
        """
        if not self.enabled:
            return False
        
        try:
            for name, value in scores.items():
                self.client.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                )
            return True
        except Exception as e:
            print(f"Error adding scores to Langfuse: {e}")
            return False
    
    def get_trace_url(self, trace_id: str) -> Optional[str]:
        """
        Generate a Langfuse trace URL.
        
        Args:
            trace_id (str): Langfuse trace ID
            
        Returns:
            Optional[str]: Full URL to view the trace in Langfuse UI
            
        Example:
            url = tracer.get_trace_url("trace-abc123")
            # Returns: "https://cloud.langfuse.com/trace/trace-abc123"
        """
        if not self.enabled or not trace_id:
            return None
        
        # Remove protocol if present in host for consistent URL building
        host = self.host.rstrip("/")
        return f"{host}/trace/{trace_id}"
    
    def flush(self):
        """
        Flush pending traces to Langfuse.
        
        Call this before application shutdown to ensure all traces are sent.
        """
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                print(f"Error flushing Langfuse: {e}")


# Global tracer instance (singleton pattern)
_tracer_instance: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """
    Get the global Langfuse tracer instance.
    
    Uses singleton pattern to ensure only one tracer is created.
    
    Returns:
        LangfuseTracer: Global tracer instance
        
    Usage:
        from lib.observability.langfuse_cb import get_tracer
        
        tracer = get_tracer()
        callback = tracer.get_callback_handler(session_id="thread-123")
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = LangfuseTracer()
    return _tracer_instance


def create_trace_context(
    session_id: str,
    user_id: str,
    platform: str,
    topic: str,
) -> Dict[str, Any]:
    """
    Create a trace context dictionary for Langfuse.
    
    Convenience function to build metadata for tracing.
    
    Args:
        session_id (str): Session/thread ID
        user_id (str): User ID
        platform (str): Platform (linkedin, twitter, etc.)
        topic (str): Post topic
        
    Returns:
        Dict[str, Any]: Metadata dictionary for tracing
    """
    return {
        "session_id": session_id,
        "user_id": user_id,
        "platform": platform,
        "topic": topic,
        "application": "agentic-post-gen",
    }


if __name__ == "__main__":
    # Test Langfuse connection
    print("Testing Langfuse connection...")
    tracer = LangfuseTracer()
    
    if tracer.enabled:
        print("Langfuse is configured and ready")
        print(f"   Host: {tracer.host}")
        
        # Test callback creation
        handler = tracer.get_callback_handler(
            session_id="test-session",
            user_id="test-user",
            metadata={"test": True}
        )
        if handler:
            print("Callback handler created successfully")
    else:
        print("Langfuse is not configured")
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing")
