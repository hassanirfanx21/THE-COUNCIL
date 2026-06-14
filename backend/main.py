from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid
from dotenv import load_dotenv

load_dotenv()  # ← reads backend/.env automatically

# ── LangGraph imports ──────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from psycopg_pool import ConnectionPool

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
except ImportError:
    from langgraph_checkpoint_postgres import PostgresSaver
    from langgraph_checkpoint_postgres import PostgresStore

from langchain_core.messages import HumanMessage, AIMessage

# ──────────────────────────────────────────────────────────────
# CONFIG — set via env vars in production
# ──────────────────────────────────────────────────────────────
DB_URI   = os.getenv("DB_URI",   "postgresql://langchain:langchain@localhost:5432/langgraph_mem")
GROQ_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_here")
SSL_MODE = os.getenv("SSL_MODE", "disable")   # "require" for Supabase

# ──────────────────────────────────────────────────────────────
# SETUP (runs once on startup)
# ──────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

conn_string = f"{DB_URI}?sslmode={SSL_MODE}"
pool = ConnectionPool(conninfo=conn_string, max_size=10, kwargs={"autocommit": True})

memory = PostgresSaver(pool)
memory.setup()

store = PostgresStore(pool, index={"dims": 768, "embed": embeddings, "fields": ["data"]})
store.setup()

# Import graph from your notebook-exported module (or inline it here)
# from council_graph import workflow   ← if you convert notebook to .py
# For now we import inline:
from council_graph import build_workflow   # ← see council_graph.py

workflow = build_workflow(pool, memory, store, GROQ_KEY)

# ──────────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="The Council API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message:   str
    user_id:   str = "default_user"
    thread_id: str = "default_thread"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "user_id":   req.user_id,
        }
    }
    try:
        result = workflow.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract final AI message
    final_msg = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        "No response generated."
    )

    return {
        "decided_agent": result.get("decided_agent"),
        "agent_field":   [
            {"agent_name": aw.agent_name, "weight": aw.weight}
            for aw in (result.get("agent_field") or [])
        ],
        "agents_output": [
            {
                "name":       a.name,
                "stance":     a.stance,
                "key_point":  a.key_point,
                "memory_used":a.memory_used,
                "confidence": a.confidence,
            }
            for a in (result.get("agents_output") or [])
        ],
        "final_response": final_msg,
    }
