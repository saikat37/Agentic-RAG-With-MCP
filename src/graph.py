"""
LangGraph workflow - State definition and graph building for the Agentic RAG system.
"""
import os
import sys
import asyncio
from typing import Annotated, Sequence, Optional

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages

from agents import (
    agent,
    grade_documents,
    draft_answer,
    final_answer,
    extract_docs,
    rewrite,
    tools,
    DocumentInfo
)
from edges import route_after_grading


# -------------------
# State Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[DocumentInfo]  # Store retrieved docs with metadata
    citations: list[str]  # Store formatted citations
    draft_answer: str  # Store draft answer before citation enrichment


# -------------------
# Graph Building
# -------------------
def build_workflow() -> StateGraph:
    """Build and return the LangGraph workflow."""
    
    # Define a new graph
    workflow = StateGraph(ChatState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([tool for tool in tools])  # retrieval node with all tools
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node("draft_answer", draft_answer)  # Draft answer (parallel with extract_docs)
    workflow.add_node("extract_docs", extract_docs)  # Extract documents metadata (parallel with draft_answer)
    workflow.add_node("final_answer", final_answer)  # Final answer with citations

    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # After retrieve, use Send for parallel execution
    workflow.add_conditional_edges(
        "retrieve",
        route_after_grading,  # Returns Send objects for parallel fan-out
    )

    # Both parallel nodes converge to final_answer
    workflow.add_edge("extract_docs", "final_answer")
    workflow.add_edge("draft_answer", "final_answer")
    workflow.add_edge("final_answer", END)
    workflow.add_edge("rewrite", "agent")

    return workflow


# Build and compile the workflow
workflow = build_workflow()

# Compile graph (LangGraph Studio provides its own persistence, so no checkpointer here)
chatbot = workflow.compile()


# Helper to display graph (works in notebook or saves to file when run as script)
def show_graph(save_path="workflow_graph.png"):
    """Display the workflow graph in notebook, or save to file if in terminal."""
    png_data = chatbot.get_graph(xray=True).draw_mermaid_png()

    # Check if we're in a Jupyter/IPython notebook environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            # We're in a notebook - display inline
            from IPython.display import Image, display
            display(Image(png_data))
            return
    except (ImportError, AttributeError):
        pass

    # Not in notebook - save to file
    with open(save_path, "wb") as f:
        f.write(png_data)
    print(f"Graph saved to {save_path}")


# Simple test invocation
if __name__ == "__main__":
    # Save the workflow graph as PNG
    # show_graph()

    # Config with thread_id for memory persistence
    config = {"configurable": {"thread_id": "1"}}

    response = asyncio.run(chatbot.ainvoke(
        {
            "messages": [HumanMessage(content="what is is best way of paddy cultivation , write in consise way.")],
            "documents": [],
            "citations": [],
            "draft_answer": ""
        },
        config=config
    ))

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(response["messages"][-1].content)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS:")
    print("=" * 60)
    for doc in response.get("documents", []):
        source = os.path.basename(doc.get('source', 'Unknown'))
        page = doc.get('page')
        page_info = f" (Page {page + 1})" if page is not None else ""
        print(f"  [{doc.get('rank', 0)}] {source}{page_info}")
        print(f"      {doc.get('content', '')}...")
        print()
