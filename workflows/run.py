"""
Graph execution helpers.

Utilities for running the LangGraph workflow with observability and checkpointing.
"""

import os
from typing import Dict, Any, Optional
from workflows.build_graph import compile_graph
from workflows.state import PostGeneratorState
from lib.memory import get_checkpointer
from lib.observability import get_tracer, create_trace_context


def run_post_generator(
    user_id: str,
    session_id: str,
    topic: str,
    platform: str = "linkedin",
    tone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete post generation workflow.
    
    Orchestrates:
    1. Initialize checkpointer and tracer
    2. Build graph with checkpointing enabled
    3. Execute graph with state
    4. Collect results and trace information
    
    Args:
        user_id (str): User identifier
        session_id (str): Session/thread ID for checkpointing
        topic (str): Topic to write about
        platform (str): Target platform (default: "linkedin")
        tone (Optional[str]): Writing tone (loads from LTM if not provided)
        
    Returns:
        Dict with:
        - post_markdown (str): Final generated post
        - scores (Dict[str, float]): Quality scores
        - trace_url (Optional[str]): Langfuse trace URL
        - session_id (str): Session identifier
        - refinement_count (int): Number of refinements performed
    """
    print(f"\n{'='*60}")
    print(f"Starting post generation workflow")
    print(f"User: {user_id}")
    print(f"Session: {session_id}")
    print(f"Topic: {topic}")
    print(f"Platform: {platform}")
    print(f"{'='*60}\n")
    
    # Initialize checkpointer
    checkpointer = get_checkpointer()
    
    # Initialize tracer
    tracer = get_tracer()
    
    # Create trace context
    trace_metadata = create_trace_context(
        session_id=session_id,
        user_id=user_id,
        platform=platform,
        topic=topic,
    )
    
    # Get callback handler for tracing
    callback_handler = tracer.get_callback_handler(
        session_id=session_id,
        user_id=user_id,
        metadata=trace_metadata,
    )
    
    # Compile graph with checkpointer
    graph = compile_graph(checkpointer=checkpointer)
    
    # Prepare initial state
    initial_state: PostGeneratorState = {
        "user_id": user_id,
        "session_id": session_id,
        "topic": topic,
        "platform": platform,
        "tone": tone,
        "refinement_count": 0,
        "max_refinements": int(os.getenv("MAX_REFINEMENT_LOOPS", "1")),
        "messages": [],
    }
    
    # Prepare config with thread_id for checkpointing
    config = {
        "configurable": {
            "thread_id": session_id,
        },
    }
    
    # Add callbacks if tracer is enabled
    if callback_handler:
        config["callbacks"] = [callback_handler]
    
    try:
        # Execute graph
        print("Executing graph...\n")
        final_state = graph.invoke(initial_state, config=config)
        
        print(f"\n{'='*60}")
        print("Workflow completed successfully")
        print(f"{'='*60}\n")
        
        # Extract results
        result = {
            "post_markdown": final_state.get("final_post", final_state.get("draft", "")),
            "scores": final_state.get("scores", {}),
            "session_id": session_id,
            "refinement_count": final_state.get("refinement_count", 0),
            "trace_url": None,
        }
        
        # Add trace URL if available
        if callback_handler and hasattr(callback_handler, "trace_id"):
            trace_id = callback_handler.trace_id
            result["trace_url"] = tracer.get_trace_url(trace_id)
            
            # Add scores to trace
            if final_state.get("scores"):
                tracer.add_scores(
                    trace_id=trace_id,
                    scores=final_state["scores"],
                    comment="DeepEval metrics from fact-checker"
                )
        
        # Flush tracer to ensure all data is sent
        tracer.flush()
        
        return result
        
    except Exception as e:
        print(f"Error during graph execution: {e}")
        tracer.flush()
        raise


def resume_session(
    session_id: str,
    user_id: str,
) -> Optional[PostGeneratorState]:
    """
    Resume a previous session from checkpoint.
    
    Args:
        session_id (str): Session ID to resume
        user_id (str): User ID (for validation)
        
    Returns:
        PostGeneratorState if found, None otherwise
    """
    checkpointer = get_checkpointer()
    
    config = {
        "configurable": {
            "thread_id": session_id,
        },
    }
    
    try:
        # Try to get checkpoint
        checkpoint = checkpointer.get_tuple(config)
        
        if checkpoint:
            print(f"Resumed session: {session_id}")
            return checkpoint
        else:
            print(f"No checkpoint found for session: {session_id}")
            return None
            
    except Exception as e:
        print(f"Error resuming session: {e}")
        return None


if __name__ == "__main__":
    # Test run
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m workflows.run <topic>")
        sys.exit(1)
    
    topic = " ".join(sys.argv[1:])
    
    result = run_post_generator(
        user_id="test-user",
        session_id=f"test-session-{topic[:20].replace(' ', '-')}",
        topic=topic,
        platform="linkedin",
        tone="insightful",
    )
    
    print("\n" + "="*60)
    print("FINAL POST:")
    print("="*60)
    print(result["post_markdown"])
    print("\n" + "="*60)
    print("SCORES:")
    print(f"  Faithfulness: {result['scores'].get('faithfulness', 'N/A')}")
    print(f"  Answer Relevancy: {result['scores'].get('answer_relevancy', 'N/A')}")
    print(f"  Refinements: {result['refinement_count']}")
    if result.get("trace_url"):
        print(f"  Trace: {result['trace_url']}")
    print("="*60)
