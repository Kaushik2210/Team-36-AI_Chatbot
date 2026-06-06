import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Import your custom standalone modules
from data_pipeline import EnterpriseDataPipeline
from agent_brain import LangGraphAgentBrain, AgentState
from memory_tools import SQLiteConversationMemory, ExternalToolsSuite

st.set_page_config(page_title="Intelligent AI Agent Workspace", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise Multi-Mode Agent Framework")

# 1. Initialize Global Core Singletons
if "pipeline" not in st.session_state: st.session_state.pipeline = EnterpriseDataPipeline()
if "memory" not in st.session_state: st.session_state.memory = SQLiteConversationMemory()
if "retriever" not in st.session_state: st.session_state.retriever = None

base_llm = ChatOllama(model="llama3", temperature=0.1, num_ctx=4096)
tools_suite = ExternalToolsSuite(base_llm)

# 2. Localized Dynamic Node linking callback for RAG
def call_rag_retrieve(state: AgentState):
    last_message = state["messages"][-1].content
    found_sources = []
    context_str = ""
    
    if st.session_state.retriever:
        docs = st.session_state.retriever.get_relevant_documents(last_message)
        context_chunks = []
        for idx, doc in enumerate(docs, 1):
            clean_source = os.path.basename(doc.metadata.get("source", "Doc")).replace("temp_", "")
            page_num = doc.metadata.get("page", 0) + 1
            context_chunks.append(f"[Source {idx}]: {doc.page_content}")
            found_sources.append({"index": idx, "file": clean_source, "page": page_num, "snippet": doc.page_content[:120] + "..."})
        context_str = "\n\n".join(context_chunks)
    else:
        context_str = "No vector space context loaded."

    contextual_prompt = f"Answer using only these document contexts.\nUse inline citations like [Source 1].\n\nContext:\n{context_str}\n\nQuery: {last_message}"
    response = base_llm.invoke([HumanMessage(content=contextual_prompt)])
    return {"messages": [response], "sources": found_sources}

# 3. Compile Master Orchestration Framework
agent_brain = LangGraphAgentBrain(web_search_fn=tools_suite.run_web_search, rag_retrieve_fn=call_rag_retrieve)

# 4. User Configuration Sidebar Interface
with st.sidebar:
    st.header("📂 Data Control Station")
    uploaded_files = st.file_uploader("Upload Company Knowledge Handbooks", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Parse & Index to FAISS Space"):
        if uploaded_files:
            with st.spinner("Processing local text partitions..."):
                os.makedirs("./temp_data", exist_ok=True)
                paths = []
                for f in uploaded_files:
                    path = f"./temp_data/temp_{f.name}"
                    with open(path, "wb") as buffer: buffer.write(f.read())
                    paths.append(path)
                st.session_state.retriever = st.session_state.pipeline.build_vector_store(paths)
                st.success("Vector DB space fully synchronized online!")
        else:
            st.warning("Please upload files first.")
            
    if st.button("Purge Application Memory Logs"):
        st.session_state.memory.clear_all_memory()
        st.rerun()

# 5. Core Chat Message Interface Loop
history = st.session_state.memory.load_history()
for msg in history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

if user_input := st.chat_input("Enter your request here..."):
    with st.chat_message("user"): st.write(user_input)
    st.session_state.memory.save_message("user", user_input)
    
    with st.chat_message("assistant"):
        status_widget = st.status("Graph execution running inference layers...", expanded=True)
        
        current_history = st.session_state.memory.load_history()
        output = agent_brain.graph.invoke({"messages": current_history, "sources": []})
        
        final_answer = output["messages"][-1].content
        retrieved_sources = output.get("sources", [])
        
        status_widget.update(label="Response finalized by dynamic node pathing!", state="complete", expanded=False)
        st.write(final_answer)
        
        if retrieved_sources:
            with st.expander("🔍 View Verified Document Citations"):
                for src in retrieved_sources:
                    st.markdown(f"**[Source {src['index']}]** `{src['file']}` — Page {src['page']}")
                    st.caption(f"*Snippet:* {src['snippet']}")
                    
        st.session_state.memory.save_message("assistant", final_answer)