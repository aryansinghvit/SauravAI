from typing import Annotated, TypedDict
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of the Progressive Context Agent.
    This object is passed between all nodes in the LangGraph.
    """
    
    # 1. Standard Conversation Memory
    # The 'add_messages' reducer automatically appends new messages to the history.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 2. Context Tracking (To prevent redundant tool calls)
    # We track the IDs of what we've fetched so the agent knows what it already knows.
    has_index: bool                                       # True if the agent has seen the topic list
    fetched_summaries: Annotated[list[str], operator.add] # e.g., ["authentication", "troubleshooting"]
    fetched_details: Annotated[list[str], operator.add]   # e.g., ["authentication/oauth2_flow"]
    
    # 3. The Working Memory (The actual text injected into the LLM prompt)
    # We append new context chunks here progressively as tools are called.
    active_context: Annotated[list[str], operator.add]
    
    # 4. Observability & Tracing (Crucial for the Streamlit UI demo)
    # We log every background action so we can show the judges exactly how the agent is "thinking"
    # Format will be a list of dicts: {"action": "Tool Call", "details": "Fetched index", "tokens": 15}
    audit_log: Annotated[list[dict], operator.add]