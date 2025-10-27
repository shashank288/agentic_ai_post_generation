# Graph Workflow (LangGraph)

The agentic workflow is implemented with LangGraph. It orchestrates four nodes in sequence with a conditional refinement loop.

- Entry → `planner` → `writer` → `fact_checker` → `router` → [refine | end]
- One refinement loop by default (configurable).

## State Schema

Defined in `workflows/state.py` as `PostGeneratorState` (TypedDict). Key fields:
- Input: `user_id`, `session_id`, `topic`, `platform`, `tone`
- Planning: `plan`, `context`, `retrieved_docs`
- Writing: `draft`, `messages`
- Fact-checking: `scores`, `needs_refinement`, `feedback`
- Control: `refinement_count`, `max_refinements`
- Output: `final_post`, `trace_id`

## Nodes

- planner (`agents/planner.py`)
  - Loads user preferences from LTM (tone fallback)
  - Retrieves context from FAISS (`lib/retriever/faiss_store.py`)
  - Produces `plan`, `context`, `retrieved_docs`

- writer (`agents/writer.py`)
  - Platform prompts (LinkedIn/Twitter)
  - Generates `draft` from `plan` + `context` (+ refinement if feedback present)
  - Appends to `messages`

- fact_checker (`agents/fact_checker.py`)
  - Computes `faithfulness` and `answer_relevancy` with DeepEval
  - Populates `scores`, `needs_refinement`, `feedback`

- router (`workflows/build_graph.py`)
  - If quality OK → sets `final_post` and ends
  - If below thresholds but loops remain → increments `refinement_count` and routes to writer
  - If max loops reached → accepts current `draft` as `final_post`

## Conditional Logic

- Function: `should_refine(state) -> Literal["refine", "end"]`
- Conditions:
  - `needs_refinement = False` → end
  - `refinement_count >= max_refinements` → end
  - else → refine

## Configuration

- `MAX_REFINEMENT_LOOPS` (default: 1)
- `FAITHFULNESS_THRESHOLD` and `ANSWER_RELEVANCY_THRESHOLD` (default: 0.80)

## Running the Graph

Via API:
- POST `/sessions` → get `session_id`
- POST `/posts:generate` with `session_id`, `topic`, `platform`, `tone`

## Observability

- If Langfuse is configured (`LANGFUSE_HOST`, keys in `.env`), the graph attaches a callback handler and records spans and scores.

## Error Handling
- Missing FAISS index → retriever raises `ValueError` (API returns 424)
- DeepEval unavailable → uses mock scores; loop still functions
- Exceptions are logged and bubbled to API with standard error body
