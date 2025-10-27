"""
LangGraph workflow construction.

Builds the multi-agent graph: Planner → Writer → Fact-Checker → Router → (conditional) END
"""

import os
from typing import Literal
from langgraph.graph import StateGraph, END

from agents import create_plan, write_post, check_facts
from workflows.state import PostGeneratorState


def router_node(state: PostGeneratorState) -> PostGeneratorState:
    """
    Router node that decides whether to refine or finish.
    
    Logic:
    - If scores meet thresholds → mark as final and go to END
    - If scores below threshold AND refinements remaining → increment count and loop to writer
    - If max refinements reached → accept current draft and go to END
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with routing decision
    """
    print("ROUTER: Deciding next step...")
    
    needs_refinement = state.get("needs_refinement", False)
    refinement_count = state.get("refinement_count", 0)
    max_refinements = state.get("max_refinements", int(os.getenv("MAX_REFINEMENT_LOOPS", "1")))
    
    if not needs_refinement:
        # Quality thresholds met
        print("Proceeding to finalize post")
        state["final_post"] = state.get("draft", "")
        return state
    
    if refinement_count >= max_refinements:
        # Max refinements reached
        print(f"Max refinements ({max_refinements}) reached. Accepting current draft.")
        state["final_post"] = state.get("draft", "")
        state["needs_refinement"] = False  # Override to prevent further loops
        return state
    
    # Refinement needed and available
    print(f"Routing back to writer for refinement #{refinement_count + 1}")
    state["refinement_count"] = refinement_count + 1
    return state


def should_refine(state: PostGeneratorState) -> Literal["refine", "end"]:
    """
    Conditional edge function to route after fact-checking.
    
    Args:
        state: Current graph state
        
    Returns:
        "refine" to loop back to writer, or "end" to finish
    """
    needs_refinement = state.get("needs_refinement", False)
    refinement_count = state.get("refinement_count", 0)
    max_refinements = state.get("max_refinements", int(os.getenv("MAX_REFINEMENT_LOOPS", "1")))
    
    # Route to end if quality is good OR max refinements reached
    if not needs_refinement or refinement_count >= max_refinements:
        return "end"
    
    return "refine"


def build_graph() -> StateGraph:
    """
    Construct the LangGraph workflow.
    
    Graph structure:
    START → planner → writer → fact_checker → router → [conditional]
                                                          ├─ (refine) → writer
                                                          └─ (end) → END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with state schema
    workflow = StateGraph(PostGeneratorState)
    
    # Add nodes
    workflow.add_node("planner", create_plan)
    workflow.add_node("writer", write_post)
    workflow.add_node("fact_checker", check_facts)
    workflow.add_node("router", router_node)
    
    # Add edges
    # Linear flow: planner → writer → fact_checker → router
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "router")
    
    # Conditional edge from router
    workflow.add_conditional_edges(
        "router",
        should_refine,
        {
            "refine": "writer",  # Loop back to writer
            "end": END,          # Finish
        }
    )
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    return workflow


def compile_graph(checkpointer=None):
    """
    Compile the graph with optional checkpointer.
    
    Args:
        checkpointer: Optional checkpointer (MemorySaver or CosmosCheckpointer)
        
    Returns:
        Compiled graph ready for invocation
    """
    workflow = build_graph()
    
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


if __name__ == "__main__":
    # Test graph construction
    print("Building LangGraph workflow...")
    graph = build_graph()
    print("Graph structure:")
    print(f"Nodes: {list(graph.nodes.keys())}")
    print(f"Entry point: planner")
    print(f"Conditional routing: router → [refine, end]")
