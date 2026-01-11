"""
Agent functions, LLM initialization, and MCP tools for the Agentic RAG system.
"""
import os
import sys
import asyncio
import json
import threading
from typing import Literal, Optional, Annotated, Sequence
from dotenv import load_dotenv

# Ensure project root (parent of `src`) is on `sys.path` so sibling packages
# like `exception`, `logger`, and `utils` are importable.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils.model_loader import ModelLoader
from exception.custom_exception import CustomException
from logger import GLOBAL_LOGGER as log
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# -----------------------------
# LLM Initialization
# -----------------------------
llm = ModelLoader().load_llm()

# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview",
#     temperature=0.3,  
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )  

# -----------------------------
# Async Loop for Backend Tasks
# -----------------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -----------------------------
# MCP Client & Tools
# -----------------------------
client = MultiServerMCPClient(
    {
        "document": {
            "transport": "http",
            "url": "https://rag-mcp-tools.fastmcp.app/mcp"
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()
tools = mcp_tools
llm_with_tools = llm.bind_tools(tools) if tools else llm

print(llm_with_tools)
print(f"Loaded {len(tools)} tools from MCP and local definitions.")
print("Tools:", [tool.name for tool in tools])


# -----------------------------
# State Types
# -----------------------------
class DocumentInfo(TypedDict):
    """Represents a retrieved document with metadata."""
    rank: int
    content: str
    source: str
    page: Optional[int]
    title: Optional[str]


# -----------------------------
# Agent Node
# -----------------------------
async def agent(state):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


# -----------------------------
# Helper: Extract docs from tool response
# -----------------------------
def extract_documents_from_tool_message(tool_message: ToolMessage) -> list[DocumentInfo]:
    """Parse tool message content to extract documents with metadata."""
    documents = []
    try:
        # Handle structured content in artifact
        if hasattr(tool_message, 'artifact') and tool_message.artifact:
            artifact = tool_message.artifact
            if isinstance(artifact, dict) and 'structured_content' in artifact:
                results = artifact['structured_content'].get('results', [])
                for r in results:
                    meta = r.get('metadata', {})
                    documents.append({
                        "rank": r.get('rank', 0),
                        "content": r.get('content', ''),
                        "source": meta.get('source', 'Unknown'),
                        "page": meta.get('page', None),
                        "title": meta.get('title', None)
                    })
                return documents

        # Fallback: parse JSON from content
        content = tool_message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    content = item.get('text', '')
                    break

        if isinstance(content, str):
            data = json.loads(content)
            results = data.get('results', [])
            for r in results:
                meta = r.get('metadata', {})
                documents.append({
                    "rank": r.get('rank', 0),
                    "content": r.get('content', ''),
                    "source": meta.get('source', 'Unknown'),
                    "page": meta.get('page', None),
                    "title": meta.get('title', None)
                })
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return documents


def format_citations(documents: list[DocumentInfo]) -> list[str]:
    """Format documents into citation strings."""
    citations = []
    seen_sources = set()
    for doc in documents:
        source = doc.get('source', 'Unknown')
        if source in seen_sources:
            continue
        seen_sources.add(source)

        # Extract filename from path
        filename = os.path.basename(source) if source != 'Unknown' else 'Unknown Source'
        title = doc.get('title') or filename
        page = doc.get('page')

        if page is not None:
            citations.append(f"{title} (Page {page + 1})")
        else:
            citations.append(title)
    return citations


def extract_docs(state):
    """
    After tool node runs, extract documents from tool messages and store in state.
    This runs in parallel with draft_answer.
    """
    messages = state["messages"]
    documents = state.get("documents", [])

    # Find the latest tool message(s)
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            new_docs = extract_documents_from_tool_message(msg)
            if new_docs:
                documents = new_docs  # Replace with latest retrieval
                break

    citations = format_citations(documents)
    return {"documents": documents, "citations": citations}


# -----------------------------
# Grading Function
# -----------------------------
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = llm

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


# -----------------------------
# Generate / Draft Answer
# -----------------------------
def draft_answer(state):
    """
    Generate draft answer (runs in parallel with extract_docs).

    Args:
        state (messages): The current state

    Returns:
         dict: The draft answer stored in state
    """
    print("---DRAFT ANSWER---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Question: {question}
Context: {context}

Answer:""",
        input_variables=["context", "question"],
    )

    # Chain using global llm
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"draft_answer": response}


def final_answer(state):
    """
    Take draft answer + documents and produce final answer with inline citations.
    """
    print("---FINAL ANSWER WITH CITATIONS---")

    draft = state.get("draft_answer", "")
    documents = state.get("documents", [])
    citations = state.get("citations", [])
    question = state["messages"][0].content

    # Build source context
    sources_info = ""
    if documents:
        sources_info = "\n\nAvailable Sources:\n"
        for i, doc in enumerate(documents, 1):
            source_name = os.path.basename(doc.get('source', 'Unknown'))
            title = doc.get('title') or source_name
            page = doc.get('page')
            page_info = f" (Page {page + 1})" if page is not None else ""
            sources_info += f"[{i}] {title}{page_info}\n"
            sources_info += f"    Content: {doc.get('content', '')[:200]}...\n\n"

    # Prompt to add natural citations
    prompt = PromptTemplate(
        template="""You are a helpful assistant. You have a draft answer and source documents. Your task is to enhance the draft answer with natural, human-like inline citations.

Rules:
- Add citations naturally using phrases like "According to [Source]...", "As mentioned in...", "The document states that..."
- Don't cite every sentence - only cite specific claims or facts
- Keep the answer fluent and readable
- At the end, add a brief "Sources:" section listing references used

Draft Answer:
{draft_answer}

Source Documents:
{sources_info}

Original Question: {question}

Provide the enhanced answer with natural inline citations:""",
        input_variables=["draft_answer", "sources_info", "question"],
    )

    # Chain using global llm
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "draft_answer": draft,
        "sources_info": sources_info,
        "question": question
    })

    return {"messages": [response]}


def generate(state):
    """
    Generate answer using retrieved documents (legacy function for compatibility).

    Args:
        state (dict): The current state with messages

    Returns:
        dict: The updated state with the generated answer
    """
    return draft_answer(state)


# -----------------------------
# Rewrite Query
# -----------------------------
def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = llm
    response = model.invoke(msg)
    return {"messages": [response]}
