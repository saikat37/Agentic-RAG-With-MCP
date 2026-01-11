"""
Edge functions and routing logic for the Agentic RAG workflow.
"""
from langgraph.types import Send
from agents import grade_documents


def route_after_grading(state) -> list[Send]:
    """
    Route to parallel nodes (extract_docs + draft_answer) if relevant,
    or to rewrite if not relevant.
    
    This function is called after the retrieve node to determine the next step.
    If documents are relevant, it fans out to both extract_docs and draft_answer
    nodes in parallel. If not relevant, it routes to the rewrite node.
    
    Args:
        state: The current graph state containing messages and retrieved documents
        
    Returns:
        list[Send]: A list of Send objects directing flow to the next node(s)
    """
    decision = grade_documents(state)
    if decision == "generate":
        # Fan-out: run both nodes in parallel
        return [Send("extract_docs", state), Send("draft_answer", state)]
    else:
        return [Send("rewrite", state)]
