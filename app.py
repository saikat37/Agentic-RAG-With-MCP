"""
Streamlit Chat Interface for Agentic RAG System
"""
import os
import sys
import asyncio

# Ensure project root and src are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import chatbot from graph
from src.graph import chatbot

# Page config
st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Chat")
st.markdown("Ask questions and get answers with citations from retrieved documents.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None

# Sidebar for state inspection
with st.sidebar:
    st.header("📊 State Inspector")
    
    if st.session_state.last_state:
        state = st.session_state.last_state
        
        # Documents section
        st.subheader("📄 Retrieved Documents")
        documents = state.get("documents", [])
        if documents:
            for i, doc in enumerate(documents, 1):
                with st.expander(f"Document {i} (Rank: {doc.get('rank', 'N/A')})"):
                    st.markdown(f"**Source:** `{os.path.basename(doc.get('source', 'Unknown'))}`")
                    if doc.get('page') is not None:
                        st.markdown(f"**Page:** {doc.get('page') + 1}")
                    if doc.get('title'):
                        st.markdown(f"**Title:** {doc.get('title')}")
                    st.markdown("**Content:**")
                    st.text(doc.get('content', '')[:500] + "..." if len(doc.get('content', '')) > 500 else doc.get('content', ''))
        else:
            st.info("No documents retrieved yet.")
        
        # Citations section
        st.subheader("📝 Citations")
        citations = state.get("citations", [])
        if citations:
            for citation in citations:
                st.markdown(f"- {citation}")
        else:
            st.info("No citations yet.")
        
        # Draft Answer section
        st.subheader("✏️ Draft Answer")
        draft = state.get("draft_answer", "")
        if draft:
            with st.expander("View Draft Answer"):
                st.write(draft)
        else:
            st.info("No draft answer yet.")
        
        # Raw state
        st.subheader("🔧 Raw State")
        with st.expander("View Raw State"):
            # Show state without messages (too verbose)
            display_state = {k: v for k, v in state.items() if k != "messages"}
            st.json(display_state)
    else:
        st.info("Submit a query to see state information.")

# Main chat area
st.divider()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

# User input
user_query = st.chat_input("Ask a question...")

if user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Process with chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare initial state
                initial_state = {
                    "messages": [HumanMessage(content=user_query)],
                    "documents": [],
                    "citations": [],
                    "draft_answer": ""
                }
                
                # Config for thread persistence
                config = {"configurable": {"thread_id": "streamlit-session"}}
                
                # Run the chatbot
                response = asyncio.run(chatbot.ainvoke(initial_state, config=config))
                
                # Store full state for sidebar
                st.session_state.last_state = response
                
                # Get the final answer
                final_message = response["messages"][-1]
                if hasattr(final_message, 'content'):
                    answer = final_message.content
                else:
                    answer = str(final_message)
                
                # Display answer
                st.write(answer)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Rerun to update sidebar
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

# Footer
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_state = None
        st.rerun()
with col2:
    st.caption("Powered by LangGraph + MCP Tools")
