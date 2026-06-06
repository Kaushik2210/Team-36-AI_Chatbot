import sqlite3
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

class SQLiteConversationMemory:
    def __init__(self, db_path="conversation_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates a relational layout table to track message turn records natively."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT
                )
            ''')
            conn.commit()

    def save_message(self, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
            conn.commit()

    def load_history(self):
        history = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT role, content FROM chat_history ORDER BY id ASC")
            for role, content in cursor.fetchall():
                if role == "user":
                    history.append(HumanMessage(content=content))
                else:
                    history.append(AIMessage(content=content))
        return history

    def clear_all_memory(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()

class ExternalToolsSuite:
    def __init__(self, llm_instance):
        self.search_tool = DuckDuckGoSearchRun()
        self.llm = llm_instance

    def run_web_search(self, state):
        """Pulls text contexts live from the web and returns a clean synthesis."""
        last_message = state["messages"][-1].content
        search_results = self.search_tool.run(last_message)
        
        contextual_prompt = f"Using this live search information:\n{search_results}\nAnswer the user: {last_message}"
        response = self.llm.invoke([HumanMessage(content=contextual_prompt)])
        return {
            "messages": [response], 
            "sources": [{"index": 1, "file": "Live Web Results", "page": "N/A", "snippet": search_results[:150]}]
        }