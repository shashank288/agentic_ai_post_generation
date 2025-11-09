"""
Writer agent for generating platform-specific posts.

This agent:
1. Loads platform-specific prompt template
2. Generates draft post based on plan and tone
3. Formats output for the target platform (LinkedIn/Twitter)
4. Returns markdown-formatted post
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig


def load_writer_prompt(platform: str) -> str:
    """
    Load the platform-specific writer prompt.
    
    Args:
        platform (str): Platform name (linkedin, twitter, etc.)
        
    Returns:
        str: Prompt template for the platform
    """
    prompt_path = os.path.join("prompts", f"writer_{platform.lower()}.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback inline prompts
        if platform.lower() == "linkedin":
            return """You are a LinkedIn content creator. Write engaging, professional posts that:
- Start with a hook that grabs attention
- Use short paragraphs (2-3 lines max)
- Include relevant insights and data points
- End with a thought-provoking question or call-to-action
- Use 3-5 relevant hashtags at the end
- Target length: ~150-200 words
- Tone: {tone}

Format in markdown. Be authentic and insightful."""
        elif platform.lower() == "twitter":
            return """You are a Twitter/X content creator. Write concise, impactful posts that:
- Grab attention in the first line
- Stay under 280 characters
- Include 1-2 relevant hashtags
- Be punchy and memorable
- Tone: {tone}

Format in markdown."""
        else:
            return """You are a social media content creator. Write an engaging post for {platform} with tone: {tone}."""


def write_post(state: Dict[str, Any], *, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Writer agent node function.
    
    Generates a platform-specific post based on the plan.
    
    Args:
        state (Dict): Current graph state with plan, platform, tone, etc.
        
    Returns:
        Dict: Updated state with draft post
    """
    print(f"WRITER: Generating {state.get('platform', 'social media')} post...")
    
    # Extract inputs
    plan = state.get("plan", "")
    platform = state.get("platform", "linkedin")
    tone = state.get("tone", "professional")
    topic = state.get("topic", "")
    context = state.get("context", "")
    
    # Check if this is a refinement pass
    refinement_count = state.get("refinement_count", 0)
    feedback = state.get("feedback", "")
    
    if refinement_count > 0:
        print(f"Refinement attempt #{refinement_count}")
        print(f"Feedback: {feedback[:100]}...")
    
    # Create LLM
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.8 if refinement_count == 0 else 0.6,  # Lower temp for refinement
    )
    
    # Load platform-specific prompt
    system_prompt = load_writer_prompt(platform)
    
    # Build user message
    if refinement_count > 0:
        user_message = """Previous draft:
{previous_draft}

Feedback:
{feedback}

Outline:
{plan}

Topic: {topic}
Tone: {tone}

Research Context:
{context}

Rewrite the post addressing the feedback while maintaining quality and relevance."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_message)
        ])
        
        messages = prompt.format_messages(
            previous_draft=state.get("draft", ""),
            feedback=feedback,
            plan=plan,
            topic=topic,
            tone=tone,
            context=context,
        )
        # Always pass config to ensure callbacks are propagated
        print(f"DEBUG writer: config={config}, has callbacks={bool(config and config.get('callbacks'))}")
        response = llm.invoke(messages, config=config or {})
    else:
        user_message = """Outline:
{plan}

Topic: {topic}
Tone: {tone}

Research Context:
{context}

Write the post following the outline and using insights from the context."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_message)
        ])
        
        messages = prompt.format_messages(
            plan=plan,
            topic=topic,
            tone=tone,
            context=context,
        )
        # Always pass config to ensure callbacks are propagated
        response = llm.invoke(messages, config=config or {})
    
    draft = response.content
    
    print(f"Draft generated ({len(draft)} chars)")
    
    # Update state
    state["draft"] = draft
    state["messages"] = state.get("messages", []) + [
        {"role": "writer", "content": draft, "refinement": refinement_count}
    ]
    
    return state
