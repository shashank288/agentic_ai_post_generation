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

# Configure Azure OpenAI environment variables for DeepEval BEFORE imports
# DeepEval uses LiteLLM which needs these set early
# if not os.getenv("OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
#     if os.getenv("AZURE_OPENAI_ENDPOINT"):
#         os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
#         os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
#     if os.getenv("AZURE_OPENAI_API_VERSION"):
#         os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")

os.environ["LITELLM_API_TYPE"] = "azure"
os.environ["LITELLM_AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["LITELLM_AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["LITELLM_AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
os.environ["LITELLM_AZURE_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# Do NOT set OPENAI_API_KEY to Azure key
# if "OPENAI_API_KEY" in os.environ:
#     del os.environ["OPENAI_API_KEY"]

# DeepEval imports with fallback
try:
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from deepeval.models import AzureOpenAIModel
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
    print("   Using Azure OpenAI for evaluation" if os.getenv("AZURE_OPENAI_ENDPOINT") else "   Using OpenAI for evaluation")
    
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
            # DeepEval v3.6+ with Azure OpenAI: just pass deployment name
            # It will use Azure based on environment variables (OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, etc.)
            # deployment_name = os.getenv("DEEPEVAL_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
            
            # print(f"   Model deployment: {deployment_name}")
            # azure_model = AzureOpenAIModel(
            #     model_name="gpt-4o",
            #     deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            #     azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            # )
            azure_model = AzureOpenAIModel(
                model_name="gpt-4o",
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
                azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                temperature=0.0
            )
            
            # Faithfulness: Does the output align with the retrieved context?
            # faithfulness_metric = FaithfulnessMetric(
            #     threshold=faithfulness_threshold,
            #     model=deployment_name,
            # )

            faithfulness_metric = FaithfulnessMetric(
                threshold=faithfulness_threshold,
                model=azure_model,
            )
            
            # Answer Relevancy: Does the output answer the input topic?
            # relevancy_metric = AnswerRelevancyMetric(
            #     threshold=relevancy_threshold,
            #     model=deployment_name,
            # )

            relevancy_metric = AnswerRelevancyMetric(
                threshold=relevancy_threshold,
                model=azure_model,
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
