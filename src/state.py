from typing import Annotated, TypedDict
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of the Progressive Context Agent.
    This object is passed between all nodes in the LangGraph.
    """
    
    # 1. saurav bhai ye normal conversation memory hai
    # 'add_messages' reducer apne aap naye messages ko history me add kr deta hai saurav bhai
    messages: Annotated[list[BaseMessage], add_messages]
    
    # 2. saurav bhai yaha context track kr rhe hai taki redundant tool calls na ho
    # jo bhi fetch kiya hai uski id rakh rhe hai taki agent ko pata rahe uske paas kya hai saurav bhai
    has_index: bool                                       # True agar agent ne topic list dekh li hai saurav bhai
    fetched_summaries: Annotated[list[str], operator.add] # jaise saurav bhai ["authentication", "troubleshooting"]
    fetched_details: Annotated[list[str], operator.add]   # jaise saurav bhai ["authentication/oauth2_flow"]
    
    # 3. saurav bhai ye working memory hai (asli text jo LLM ke prompt me jata hai)
    # jaise jaise tools call hote hai waise waise naye context chunks yaha add krte jate hai saurav bhai
    active_context: Annotated[list[str], operator.add]
    
    # 4. saurav bhai observability aur tracing ke liye (Streamlit UI demo me bahut zaroori hai)
    # har background action log kr rhe hai taki judges ko dikha sake agent kaise "soch" raha hai saurav bhai
    # format aisa rahega saurav bhai: {"action": "Tool Call", "details": "Fetched index", "tokens": 15}
    audit_log: Annotated[list[dict], operator.add]