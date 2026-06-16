from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from council_graph import build_workflow
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="The Council API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build once on startup — lives in RAM
workflow = build_workflow(groq_key=os.getenv("GROQ_API_KEY", "your-groq-key"))

class ChatRequest(BaseModel):
    message:   str
    thread_id: str = "default_thread"
    user_id:   str = "default_user"

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "user_id":   req.user_id
        }
    }
    input_data = {"messages": [HumanMessage(content=req.message)]}
    
    try:
        response = workflow.invoke(input_data, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Return exactly what the React frontend expects
    return {
        "decided_agent": response.get("decided_agent"),
        "agent_field":   [
            {"agent_name": getattr(aw, 'agent_name', ''), "weight": getattr(aw, 'weight', 0.25)}
            for aw in (response.get("agent_field") or [])
        ],
        "agents_output": [
            {
                "name":       getattr(a, 'name', ''),
                "stance":     getattr(a, 'stance', ''),
                "key_point":  getattr(a, 'key_point', ''),
                "memory_used":getattr(a, 'memory_used', []),
                "confidence": getattr(a, 'confidence', 0.0),
            }
            for a in (response.get("agents_output") or [])
        ],
        "final_response": response["messages"][-1].content,
    }
