import os
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Initialize Local LLM Brain
llm = ChatOllama(model="llama3", temperature=0.1, num_ctx=4096)
web_search_tool = DuckDuckGoSearchRun()

# 2. Define State Workflow Core
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    router_decision: str
    sources: List[dict]

# 3. Intelligent Supervisor Router Node
def intent_router(state: AgentState):
    last_message = state["messages"][-1].content
    
    system_prompt = (
        "You are an routing engineer. Classify the user query into exactly one of three categories:\n"
        "1. 'web_search' - If the query asks for real-time data, current events, or live updates.\n"
        "2. 'rag_retrieve' - If the query asks for organizational data, internal policies, or uploaded metrics.\n"
        "3. 'llm_chat' - For general knowledge, coding, or conversational text processing.\n"
        "Respond with exactly one word from options: ['web_search', 'rag_retrieve', 'llm_chat']."
    )
    
    decision = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    parsed_decision = decision.content.strip().lower()
    
    if "web_search" in parsed_decision:
        return "web_search"
    elif "rag_retrieve" in parsed_decision:
        return "rag_retrieve"
    else:
        return "llm_chat"

# 4. Action Node Logic
def call_llm_chat(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response], "sources": []}

def call_web_search(state: AgentState):
    last_message = state["messages"][-1].content
    search_results = web_search_tool.run(last_message)
    
    contextual_prompt = f"Using this real-time web information:\n{search_results}\nAnswer the user: {last_message}"
    response = llm.invoke([HumanMessage(content=contextual_prompt)])
    return {"messages": [response], "sources": [{"index": 1, "file": "Live Web Search", "page": "N/A", "snippet": search_results[:150]}]}

def call_rag_retrieve(state: AgentState):
    last_message = state["messages"][-1].content
    global global_retriever
    
    found_sources = []
    context_str = ""
    
    if 'global_retriever' in globals() and global_retriever:
        docs = global_retriever.get_relevant_documents(last_message)
        context_chunks = []
        for idx, doc in enumerate(docs, 1):
            clean_source = os.path.basename(doc.metadata.get("source", "Doc")).replace("temp_", "")
            page_num = doc.metadata.get("page", 0) + 1
            context_chunks.append(f"[Source {idx}]: {doc.page_content}")
            
            found_sources.append({
                "index": idx,
                "file": clean_source,
                "page": page_num,
                "snippet": doc.page_content[:150] + "..."
            })
        context_str = "\n\n".join(context_chunks)
    else:
        context_str = "No local knowledge bases compiled or active."

    contextual_prompt = (
        f"Answer the user query using only the provided contexts.\n"
        f"You MUST use inline citations matching the source block numbering (e.g. [Source 1]).\n\n"
        f"Context:\n{context_str}\n\nQuery: {last_message}"
    )
    response = llm.invoke([HumanMessage(content=contextual_prompt)])
    return {"messages": [response], "sources": found_sources}

# 5. Graph compilation
workflow = StateGraph(AgentState)
workflow.add_node("llm_chat", call_llm_chat)
workflow.add_node("web_search", call_web_search)
workflow.add_node("rag_retrieve", call_rag_retrieve)

workflow.set_conditional_entry_point(
    intent_router,
    {"llm_chat": "llm_chat", "web_search": "web_search", "rag_retrieve": "rag_retrieve"}
)
workflow.add_edge("llm_chat", END)
workflow.add_edge("web_search", END)
workflow.add_edge("rag_retrieve", END)

agent_app = workflow.compile()