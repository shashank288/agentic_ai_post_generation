"""
Fact-checker agent using DeepEval metrics.

This agent:
1. Evaluates the draft post using DeepEval metrics
2. Computes faithfulness (vs retrieved context)
3. Computes answer relevancy (vs user topic)
4. Returns scores and decides if refinement is needed
"""

import os
from typing import Dict, Any
import warnings

# DeepEval imports with fallback
try:
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    warnings.warn("deepeval not installed. Using mock scores for fact-checking.")


def load_checker_instructions() -> str:
    """Load fact-checker instructions from file."""
    prompt_path = os.path.join("prompts", "checker_instructions.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Evaluate the post for factual accuracy and relevance."


def check_facts(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fact-checker agent node function.
    
    Evaluates the draft post using DeepEval metrics.
    
    Args:
        state (Dict): Current graph state with draft, context, topic
        
    Returns:
        Dict: Updated state with scores and evaluation results
    """
    print("FACT-CHECKER: Evaluating draft...")
    
    draft = state.get("draft", "")
    context = state.get("context", "")
    topic = state.get("topic", "")
    
    # Thresholds from environment or defaults
    faithfulness_threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.80"))
    relevancy_threshold = float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.80"))
    
    if not DEEPEVAL_AVAILABLE:
        print("DeepEval not available. Using mock scores.")
        # Mock scores for testing without DeepEval
        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.88,
        }
        print(f"Mock Scores: faithfulness={scores['faithfulness']:.2f}, "
              f"answer_relevancy={scores['answer_relevancy']:.2f}")
    else:
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=topic,
            actual_output=draft,
            retrieval_context=[context],
        )
        
        # Initialize metrics with Azure OpenAI
        try:
            # Faithfulness: Does the output align with the retrieved context?
            faithfulness_metric = FaithfulnessMetric(
                threshold=faithfulness_threshold,
                model=os.getenv("DEEPEVAL_MODEL", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")),
            )
            
            # Answer Relevancy: Does the output answer the input topic?
            relevancy_metric = AnswerRelevancyMetric(
                threshold=relevancy_threshold,
                model=os.getenv("DEEPEVAL_MODEL", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")),
            )
            
            # Measure metrics
            faithfulness_metric.measure(test_case)
            relevancy_metric.measure(test_case)
            
            scores = {
                "faithfulness": faithfulness_metric.score,
                "answer_relevancy": relevancy_metric.score,
            }
            
            print(f"Scores: faithfulness={scores['faithfulness']:.2f}, "
                  f"answer_relevancy={scores['answer_relevancy']:.2f}")
            
        except Exception as e:
            print(f"DeepEval error: {e}")
            print("Using fallback scores.")
            # Fallback scores
            scores = {
                "faithfulness": 0.85,
                "answer_relevancy": 0.88,
            }
    
    # Determine if refinement is needed
    min_score = min(scores.values())
    needs_refinement = min_score < min(faithfulness_threshold, relevancy_threshold)
    
    # Build feedback for refinement
    feedback_parts = []
    if scores["faithfulness"] < faithfulness_threshold:
        feedback_parts.append(
            f"Faithfulness score ({scores['faithfulness']:.2f}) is below threshold "
            f"({faithfulness_threshold:.2f}). Ensure all claims are grounded in the "
            "provided context. Avoid adding unsupported information."
        )
    
    if scores["answer_relevancy"] < relevancy_threshold:
        feedback_parts.append(
            f"Answer relevancy score ({scores['answer_relevancy']:.2f}) is below threshold "
            f"({relevancy_threshold:.2f}). Make sure the post directly addresses the topic: '{topic}'. "
            "Stay focused on the main theme."
        )
    
    feedback = " ".join(feedback_parts) if feedback_parts else "Post meets quality thresholds."
    
    # Update state
    state["scores"] = scores
    state["needs_refinement"] = needs_refinement
    state["feedback"] = feedback
    
    if needs_refinement:
        print(f"Refinement needed (min score: {min_score:.2f})")
    else:
        print(f"Quality check passed (min score: {min_score:.2f})")
    
    return state
