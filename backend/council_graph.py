from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver      # in-memory STM
from langgraph.store.memory import InMemoryStore         # in-memory LTM
from pydantic import BaseModel, Field, model_validator
from typing import TypedDict, Annotated, Literal, List
import uuid

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
except ImportError:
    PostgresSaver = None
    PostgresStore = None


# ─── Schemas ──────────────────────────────────────────────────────────────────

class agent_weight(BaseModel):
    agent_name: Literal["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN", "UNDECIDED"]
    weight: float = Field(ge=0.0, le=1.0)

class AgentOutput(BaseModel):
    name:        Literal["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN"]
    stance:      str
    key_point:   str
    memory_used: List[str]
    confidence:  float

class CouncilFullOutput(BaseModel):
    """
    Single structured output schema.
    One LLM call fills classification + all agent perspectives + summary + response.
    """
    decided_agent:   Literal["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN", "UNDECIDED"]
    agent_field:     List[agent_weight]
    agents_output:   List[AgentOutput]
    updated_summary: str
    final_response:  str

    @model_validator(mode="after")
    def normalize_weights(self) -> "CouncilFullOutput":
        total = sum(a.weight for a in self.agent_field)
        if total > 0 and abs(total - 1.0) > 1e-4:
            for a in self.agent_field:
                a.weight = round(a.weight / total, 4)
            # fix any remaining rounding drift on the first item
            drift = round(1.0 - sum(a.weight for a in self.agent_field), 4)
            self.agent_field[0].weight = round(self.agent_field[0].weight + drift, 4)
        return self


# ─── Custom reducer ───────────────────────────────────────────────────────────

def add_or_reset(existing: list, new: list | None) -> list:
    """None resets the list. Otherwise append only new names."""
    if new is None:
        return []
    existing_names = {a.name for a in existing}
    return existing + [a for a in new if a.name not in existing_names]


# ─── State ────────────────────────────────────────────────────────────────────

class chat_state(TypedDict):
    messages:       Annotated[list, add_messages]
    summary:        str
    decided_agent:  Literal["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN", "UNDECIDED"]
    agent_field:    List[agent_weight]
    agent_memories: list
    agents_output:  Annotated[list, add_or_reset]


# ─── build_workflow ───────────────────────────────────────────────────────────
#
# Two ways to call this:
#
#   LOCAL (no Postgres — data lost on restart):
#     workflow = build_workflow(groq_key="...")
#
#   PRODUCTION (Postgres — data persists):
#     workflow = build_workflow(groq_key="...", memory=pg_saver, store=pg_store)
#
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow(groq_key: str, memory=None, store=None, pool=None):
    """
    memory : checkpointer for STM  — defaults to MemorySaver  (in-RAM)
    store  : key-value store for LTM — defaults to InMemoryStore (in-RAM)
    pool   : only needed when passing a PostgresSaver/PostgresStore pair
    """
    if memory is None:
        memory = MemorySaver()        # STM lives in RAM

    if store is None:
        store = InMemoryStore()       # LTM lives in RAM

    llm = ChatGroq(
        api_key=groq_key,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_retries=3
    )

    # ── STM_Loader ── NO LLM ──────────────────────────────────────────────────
    def STM_Loader(state: chat_state):
        """
        Trims old messages from state — keeps last 2.
        Resets agents_output to [] so Synthesizer can fill it fresh.
        No LLM call.
        """
        if len(state["messages"]) > 2:
            to_delete = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
            return {"agents_output": None, "messages": to_delete}
        return {"agents_output": None}

    # ── Problem_Classifier ── NO LLM ─────────────────────────────────────────
    def Problem_Classifier(state: chat_state):
        """
        Returns placeholder defaults.
        Synthesizer will overwrite decided_agent and agent_field with real values.
        No LLM call.
        """
        return {
            "decided_agent": "UNDECIDED",
            "agent_field": [
                agent_weight(agent_name="STRATEGIST",  weight=0.25),
                agent_weight(agent_name="REALIST",     weight=0.25),
                agent_weight(agent_name="ADVOCATE",    weight=0.25),
                agent_weight(agent_name="CONTRARIAN",  weight=0.25),
            ]
        }

    # ── Memory_Fetcher ── NO LLM ─────────────────────────────────────────────
    def Memory_fetcher(state: chat_state, config: RunnableConfig):
        """
        Vector search only — no LLM needed.
        Uses the last user message as the query and distributes results across agents.
        """
        user_id   = config["configurable"].get("user_id", "default_user")
        namespace = ("users", user_id, "memories")

        try:
            query = state["messages"][-1].content if state["messages"] else ""
            raw   = store.search(namespace, query=query, limit=12)
            agents = ["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN"]
            # round-robin distribute memories across agents
            result = [
                {"agent_name": a, "memories": raw[i::4]}
                for i, a in enumerate(agents)
            ]
        except Exception:
            result = [
                {"agent_name": a, "memories": []}
                for a in ["STRATEGIST", "REALIST", "ADVOCATE", "CONTRARIAN"]
            ]

        return {"agent_memories": result}

    # ── Synthesizer ── THE ONE REAL LLM CALL ─────────────────────────────────
    def Synthesizer(state: chat_state, config: RunnableConfig):
        """
        Single LLM call that does everything:
          - Classifies the problem (decided_agent + agent_field weights)
          - Generates all four agent perspectives (agents_output)
          - Updates the user profile summary
          - Produces the final synthesized response
        """
        summary = state.get("summary", "No prior summary.")

        memories_flat = [
            item.value["data"]
            for ag in state.get("agent_memories", [])
            for item in ag.get("memories", [])
            if hasattr(item, "value") and isinstance(item.value, dict)
        ]

        PROMPT = f"""You are a council of four advisors and a synthesizer combined into one call.

User Profile Summary:
{summary}

Relevant memories from past conversations:
{memories_flat[:6] if memories_flat else "None yet."}

Given the user's latest message, produce a structured output covering all of the following:

1. decided_agent
   Which single agent type best represents the core of this problem:
   STRATEGIST | REALIST | ADVOCATE | CONTRARIAN | UNDECIDED

2. agent_field
   Weights for all four agents (must sum exactly to 1.0):
   - STRATEGIST  long-term goals and life trajectory
   - REALIST     hard constraints, risks, documented patterns
   - ADVOCATE    what the user genuinely wants and values
   - CONTRARIAN  the blind spot or ignored angle

3. agents_output
   All four agents each providing:
   - stance:      their overall position in 1-2 sentences
   - key_point:   the single most important thing they want said
   - memory_used: list of any memories drawn on (can be empty list)
   - confidence:  float 0.0 to 1.0

4. updated_summary
   Merge the existing user profile with any new information from the latest messages.
   150-250 words. Structured paragraphs. Factual. No repetition.

5. final_response
   One unified voice that synthesizes all four perspectives.
   Speak directly to the user. Do NOT frame it as "Strategist said / Realist said".
   End with one forward-moving question.
"""

        council_llm = llm.with_structured_output(CouncilFullOutput)
        msgs        = [SystemMessage(content=PROMPT)] + state["messages"]
        resp        = council_llm.invoke(msgs)

        return {
            "decided_agent": resp.decided_agent,
            "agent_field":   resp.agent_field,
            "agents_output": resp.agents_output,
            "summary":       resp.updated_summary,
            "messages":      [AIMessage(content=resp.final_response)],
        }

    # ── Memory_Writer ── NO LLM ───────────────────────────────────────────────
    def Memory_writer(state: chat_state, config: RunnableConfig):
        """
        Stores the last exchange directly into the vector store.
        Skips if a near-duplicate already exists (score > 0.75).
        No LLM call.
        """
        user_id   = config["configurable"].get("user_id", "default_user")
        namespace = ("users", user_id, "memories")

        last_msgs = [m for m in state["messages"][-2:] if hasattr(m, "content")]
        if not last_msgs:
            return

        combined = " | ".join(m.content[:300] for m in last_msgs)

        try:
            existing = store.search(namespace, query=combined, limit=3)
            for e in existing:
                if e.score > 0.75:
                    return  # near-duplicate — skip
            store.put(namespace, str(uuid.uuid4()), {"data": combined})
        except Exception:
            pass

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph = StateGraph(chat_state)

    graph.add_node("STM_Loader",         STM_Loader)
    graph.add_node("Problem_Classifier", Problem_Classifier)
    graph.add_node("Memory_fetcher",     Memory_fetcher)
    graph.add_node("Synthesizer",        Synthesizer)
    graph.add_node("Memory_Writer",      Memory_writer)

    graph.add_edge(START,                "STM_Loader")
    graph.add_edge("STM_Loader",         "Problem_Classifier")
    graph.add_edge("Problem_Classifier", "Memory_fetcher")
    graph.add_edge("Memory_fetcher",     "Synthesizer")
    graph.add_edge("Synthesizer",        "Memory_Writer")
    graph.add_edge("Memory_Writer",      END)

    return graph.compile(checkpointer=memory, store=store)