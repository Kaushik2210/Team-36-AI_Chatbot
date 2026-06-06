import os
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# =======================================================
# 1. High-Efficiency Stateful Schema
# =======================================================
class AgentState(TypedDict):
    """
    Optimized state window. Instead of passing massive raw documents down the graph,
    we pass a streamlined 'sources' metadata dictionary, keeping memory footprints tiny.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    router_decision: str
    sources: List[dict]

class LangGraphAgentBrain:
    def __init__(self, web_search_fn, rag_retrieve_fn):
        # num_ctx=4096 keeps context footprint tight but sufficient. 
        # temperature=0.0 forces mathematical determinism for fast routing.
        self.llm = ChatOllama(model="llama3", temperature=0.0, num_ctx=4096)
        self.web_search_fn = web_search_fn
        self.rag_retrieve_fn = rag_retrieve_fn
        self.graph = self._compile_workflow_graph()

    # =======================================================
    # 2. Optimized Intent Routing (Data Thinning)
    # =======================================================
    def intent_router(self, state: AgentState):
        """
        EFFICIENCY KEY: Do NOT pass the entire multi-turn chat history to the LLM to route!
        We slice out only the *absolute latest message* string. This minimizes token overhead 
        by up to 80% on long conversations, speeding up local Ollama inference drastically.
        """
        last_message = state["messages"][-1].content
        
        system_prompt = (
            "You are a low-latency router. Categorize the user prompt into exactly one token string:\n"
            "1. 'web_search' - for breaking news, current dates, or real-time trends.\n"
            "2. 'rag_retrieve' - for company files, policies, or specific uploaded documents.\n"
            "3. 'llm_chat' - for logical processing, creative writing, or coding requests.\n"
            "Output only the exact token choice string. Do not include spaces, reasoning, or periods."
        )
        
        # Light inference execution
        decision = self.llm.invoke([
            SystemMessage(content=system_prompt), 
            HumanMessage(content=f"Prompt to route: {last_message}")
        ])
        
        parsed_decision = decision.content.strip().lower()
        
        # Strict fallback routing containment
        if "web_search" in parsed_decision: 
            return "web_search"
        elif "rag_retrieve" in parsed_decision: 
            return "rag_retrieve"
        else: 
            return "llm_chat"

    # =======================================================
    # 3. Message Trimming Strategy (Preventing Context Bloat)
    # =======================================================
    def call_llm_chat(self, state: AgentState):
        """
        EFFICIENCY KEY: Context window management. If your conversation history gets 
        longer than 6-10 messages, local Ollama engines slow to a crawl. We pass only 
        the last 5 messages to preserve memory context while dropping old dead weight.
        """
        trimmed_messages = state["messages"][-5:]
        response = self.llm.invoke(trimmed_messages)
        return {"messages": [response], "sources": []}

    # =======================================================
    # 4. Graph Architecture Assembly
    # =======================================================
    def _compile_workflow_graph(self):
        workflow = StateGraph(AgentState)
        
        # Register functional modular worker nodes
        workflow.add_node("llm_chat", self.call_llm_chat)
        workflow.add_node("web_search", self.web_search_fn)
        workflow.add_node("rag_retrieve", self.rag_retrieve_fn)
        
        # Inject fast conditional routing decision trees
        workflow.set_conditional_entry_point(
            self.intent_router,
            {
                "llm_chat": "llm_chat", 
                "web_search": "web_search", 
                "rag_retrieve": "rag_retrieve"
            }
        )
        
        # Establish simple linear exits to minimize graph state cycles
        workflow.add_edge("llm_chat", END)
        workflow.add_edge("web_search", END)
        workflow.add_edge("rag_retrieve", END)
        
        return workflow.compile()