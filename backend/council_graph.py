# council_graph.py
# Extract all the graph logic from your notebook here.
# This file is imported by main.py
#
# Paste everything from your notebook cells 1-9 (embeddings → graph compile) into this file.
# Then wrap it in build_workflow() so main.py can call it.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator
from typing import TypedDict, Annotated, Literal, List
import uuid, operator, json

# langgraph 1.x checkpoint/store imports
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
except ImportError:
    from langgraph_checkpoint_postgres import PostgresSaver  # fallback
    from langgraph_checkpoint_postgres import PostgresStore


# ─── Schemas ───────────────────────────────────────────────────
class agent_weight(BaseModel):
    agent_name: Literal["STRATEGIST","REALIST","ADVOCATE","CONTRARIAN","UNDECIDED"]
    weight: float = Field(ge=0.0, le=1.0)

class agent_fields_schema(BaseModel):
    agent_decision: Literal["STRATEGIST","REALIST","ADVOCATE","CONTRARIAN","UNDECIDED"]
    weight: List[agent_weight]

    @model_validator(mode="before")
    def fix_missing_agents_and_normalize(cls, values):
        if "weight" in values and isinstance(values["weight"], list):
            weights = values["weight"]
            found_names = set()
            for w in weights:
                if isinstance(w, dict) and "agent_name" in w:
                    found_names.add(w["agent_name"])
            
            for required_agent in ["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN"]:
                if required_agent not in found_names:
                    weights.append({"agent_name": required_agent, "weight": 0.0})
            
            # Normalize to exactly 1.0
            total = sum(float(w.get("weight", 0.0)) for w in weights if isinstance(w, dict))
            if total > 0:
                for w in weights:
                    if isinstance(w, dict):
                        w["weight"] = round(float(w.get("weight", 0.0)) / total, 4)
            else:
                for w in weights:
                    if isinstance(w, dict):
                        w["weight"] = 0.25
            values["weight"] = weights
        return values

    @model_validator(mode="after")
    def force_sum_to_one(self) -> "agent_fields_schema":
        # Fix any floating point rounding issues from the normalizer
        total = sum(a.weight for a in self.weight)
        if len(self.weight) > 0 and abs(total - 1.0) > 1e-4:
            self.weight[0].weight += (1.0 - total)
        return self

class agent_memories_model(BaseModel):
    agent_name: Literal["STRATEGIST","REALIST","ADVOCATE","CONTRARIAN"]
    memories: list[str]

class AgentOutput(BaseModel):
    name:        Literal["STRATEGIST","REALIST","ADVOCATE","CONTRARIAN"]
    stance:      str
    key_point:   str
    memory_used: List[str]
    confidence:  float

class LLMAgentOutput(BaseModel):
    stance:      str
    key_point:   str
    memory_used: List[str]
    confidence:  float

    @model_validator(mode="before")
    def fix_memory_used_string(cls, values):
        mem = values.get("memory_used")
        if isinstance(mem, str):
            try:
                import ast
                parsed = ast.literal_eval(mem)
                values["memory_used"] = parsed if isinstance(parsed, list) else []
            except:
                values["memory_used"] = []
        return values

class agent_query(BaseModel):
    agent_name: Literal["STRATEGIST","REALIST","ADVOCATE","CONTRARIAN"]
    query: str

class agent_query_schema(BaseModel):
    agent_queries: List[agent_query]

# ─── Custom reducer ─────────────────────────────────────────────
def add_or_reset(existing: list, new: list | None) -> list:
    if new is None:
        return []
    existing_names = {a.name for a in existing}
    return existing + [a for a in new if a.name not in existing_names]

# ─── State ──────────────────────────────────────────────────────
class chat_state(TypedDict):
    messages:      Annotated[list, add_messages]
    summary:       str
    decided_agent: Literal["STRATEGIST","REALIST","ADVOCATE","CONTRARIAN","UNDECIDED"]
    agent_field:   List[agent_weight]
    agent_memories: list
    agents_output: Annotated[list, add_or_reset]


# ─── build_workflow factory ──────────────────────────────────────
def build_workflow(pool, memory, store, groq_key: str):

    # Changed to 8b to avoid 100k TPD rate limit on the 70b model
    llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.7, max_retries=3)

    # ── STM Loader ──────────────────────────────────────────────
    def STM_Loader(state: chat_state):
        summary = state.get("summary", "")
        if summary:
            prompt = f"""You are a user profiling assistant. Merge the previous summary with new messages into an updated profile.
Previous Summary: {summary}
New Messages: {state['messages']}
Keep 200-300 words. Factual, structured."""
        else:
            prompt = f"""You are a user profiling assistant. Build an initial user profile from these messages.
Messages: {state['messages']}
Keep 150-300 words. Factual, structured."""

        new_summary = llm.invoke([HumanMessage(content=prompt)]).content

        if len(state["messages"]) > 2:
            to_delete = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
            return {"summary": new_summary, "messages": to_delete, "agents_output": None}
        return {"summary": new_summary, "agents_output": None}

    # ── Problem Classifier ──────────────────────────────────────
    def Problem_Classifier(state: chat_state):
        summary = state.get("summary", "")
        clf_llm = llm.with_structured_output(agent_fields_schema)
        PROMPT = f"""You are a routing intelligence layer. Classify the problem and weight each of the four agents.
Summary: {summary}
Agents: STRATEGIST (long-term goals), REALIST (hard truths), ADVOCATE (user's own wants), CONTRARIAN (blind spots).
Weights must sum to 1.0. If no real problem, set UNDECIDED and equal weights 0.25."""
        msgs = [SystemMessage(content=PROMPT)] + state["messages"]
        resp = clf_llm.invoke(msgs)
        return {"decided_agent": resp.agent_decision, "agent_field": resp.weight}

    # ── Memory Fetcher ──────────────────────────────────────────
    def Memory_fetcher(state: chat_state, config: RunnableConfig):
        user_id   = config["configurable"].get("user_id", "default_user")
        namespace = ("users", user_id, "memories")
        qmaker = llm.with_structured_output(agent_query_schema)
        PROMPT = """Generate 4 memory retrieval queries, one per agent (STRATEGIST, REALIST, ADVOCATE, CONTRARIAN).
Each query is a single concise semantic search line tailored to that agent's memory focus."""
        msgs = [SystemMessage(content=PROMPT)] + state["messages"]
        queries = qmaker.invoke(msgs).agent_queries

        used_keys = set()
        result = []
        for aq in queries:
            raw = store.search(namespace, query=aq.query, limit=9)
            unique = []
            for m in raw:
                if m.key not in used_keys:
                    unique.append(m)
                    used_keys.add(m.key)
                if len(unique) == 3:
                    break
            result.append({"agent_name": aq.agent_name, "memories": unique})
        return {"agent_memories": result}

    # ── Agent factory ───────────────────────────────────────────
    def make_agent(name: str, persona: str):
        def agent_fn(state: chat_state, config: RunnableConfig):
            memories = [
                item.value["data"]
                for ag in state["agent_memories"]
                if ag["agent_name"] == name
                for item in ag["memories"]
            ]
            prompt = f"""{persona}
Summary: {state['summary']}
Memories: {memories}
Respond as structured output."""
            agent_llm = llm.with_structured_output(LLMAgentOutput)
            msgs = [SystemMessage(content=prompt)] + state["messages"]
            llm_resp = agent_llm.invoke(msgs)
            
            # Reconstruct the full AgentOutput explicitly
            resp = AgentOutput(
                name=name,
                stance=llm_resp.stance,
                key_point=llm_resp.key_point,
                memory_used=llm_resp.memory_used,
                confidence=llm_resp.confidence
            )
            return {"agents_output": [resp]}
        agent_fn.__name__ = name
        return agent_fn

    Strategist = make_agent("STRATEGIST", "You are the Strategist. Connect the user's problem to their long-term goals. Calm, forward-looking, trajectory-focused.")
    Realist    = make_agent("REALIST",    "You are the Realist. Name hard constraints, real risks, and documented patterns without softening. Direct and factual.")
    Advocate   = make_agent("ADVOCATE",   "You are the Advocate. Represent what the user actually wants. Warm but not passive — push back when decisions betray their values.")
    Contrarian = make_agent("CONTRARIAN", "You are the Contrarian. Find the angle nobody is saying. Thoughtful and a little stubborn — surface what is being ignored.")

    # ── Synthesizer ─────────────────────────────────────────────
    def Synthesizer(state: chat_state, config: RunnableConfig):
        weights_str = "\n".join(
            f"  {aw.agent_name:<14} {'█' * int(aw.weight * 20):<20} {aw.weight}"
            for aw in state["agent_field"]
        )
        agents_str = "\n\n".join(
            f"[{a.name}]\nStance: {a.stance}\nKey Point: {a.key_point}\nConfidence: {a.confidence}"
            for a in state["agents_output"]
        )
        prompt = f"""You are the Synthesizer. Read across all four advisor outputs and write one honest, grounded response directly to the user.
Do NOT structure as "Strategist said... Realist said...". Speak as one voice. End with one forward-moving question.

Summary: {state['summary']}
Decided type: {state['decided_agent']}
Weights:\n{weights_str}

Advisor outputs:\n{agents_str}"""
        msgs = [SystemMessage(content=prompt)] + state["messages"]
        resp = llm.invoke(msgs)
        return {"messages": [AIMessage(content=resp.content)]}

    # ── Memory Writer ───────────────────────────────────────────
    def Memory_writer(state: chat_state, config: RunnableConfig):
        combined = ".".join([m.content for m in state["messages"][-2:]])
        user_id   = config["configurable"].get("user_id", "default_user")
        namespace = ("users", user_id, "memories")
        existing  = store.search(namespace, query=combined, limit=3)
        for e in existing:
            if e.score > 0.75:
                return
        PROMPT = """You are a memory writer. Extract one concise line of new information from this message for long-term storage. Respond concisely. Do not include internal thinking processes, chain-of-thought, or <think> tags in your final output."""
        msg = [SystemMessage(content=PROMPT)] + state["messages"]
        resp = llm.invoke(msg)
        store.put(namespace, str(uuid.uuid4()), {"data": resp.content})

    # ── Build graph ──────────────────────────────────────────────
    graph = StateGraph(chat_state)
    graph.add_node("STM_Loader",        STM_Loader)
    graph.add_node("Problem_Classifier",Problem_Classifier)
    graph.add_node("Memory_fetcher",    Memory_fetcher)
    graph.add_node("Strategist",        Strategist)
    graph.add_node("Realist",           Realist)
    graph.add_node("Advocate",          Advocate)
    graph.add_node("Contrarian",        Contrarian)
    graph.add_node("Synthesizer",       Synthesizer)
    graph.add_node("Memory_Writer",     Memory_writer)

    graph.add_edge(START,               "STM_Loader")
    graph.add_edge("STM_Loader",        "Problem_Classifier")
    graph.add_edge("Problem_Classifier","Memory_fetcher")
    graph.add_edge("Memory_fetcher",    "Strategist")
    graph.add_edge("Memory_fetcher",    "Realist")
    graph.add_edge("Memory_fetcher",    "Advocate")
    graph.add_edge("Memory_fetcher",    "Contrarian")
    graph.add_edge("Strategist",        "Synthesizer")
    graph.add_edge("Realist",           "Synthesizer")
    graph.add_edge("Advocate",          "Synthesizer")
    graph.add_edge("Contrarian",        "Synthesizer")
    graph.add_edge("Synthesizer",       "Memory_Writer")
    graph.add_edge("Memory_Writer",     END)

    return graph.compile(checkpointer=memory, store=store)
