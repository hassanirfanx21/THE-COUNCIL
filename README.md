# The Council

An agentic multi-advisor system that builds a long-term memory profile of the user to provide highly personalized, multi-perspective advisory feedback.

---

## The LLM Stack & Development History

During the initial design and construction phase, **The Council** was built and tested locally using a **local LLM (Qwen 3:4b / Qwen2.5:4b)** running on the user's local machine. This allowed for cost-effective development, quick prototyping of agent prompts, and testing of structured schema parsing.

For production use and to deliver a responsive chat experience, the backend was migrated to use **Groq** (utilizing fast open-source models such as Qwen 32B). This transition keeps the same agent architecture while drastically reducing latency from minutes to seconds.

---

## What Is This Project, In Simple Words

You have a problem. Any problem — career, decision, personal, work. You type it out like you're texting a friend.

Instead of one AI giving you one answer, **four AI advisors** — each with a completely different personality and job — read your problem simultaneously. But here is the part that makes it different from every other multi-agent demo: **these advisors have been learning about you for weeks**. They pull up what they know about you from memory before they speak. The Strategist remembers your goals. The Realist remembers your past mistakes. The Advocate remembers what you actually care about. The Contrarian remembers what you keep getting wrong.

Then a fifth node — the Synthesizer — reads all four opinions, finds where they agree and where they clash, and gives you one final clear response that feels like it came from someone who genuinely knows you.

Every session, new things you reveal get stored back into memory. Over time the system builds a deep model of who you are. The longer you use it, the smarter the advice gets.

---

## The Four Advisors — Who They Are

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   THE STRATEGIST              THE REALIST                               │
│   ─────────────────           ─────────────────                         │
│   Thinks 2 years ahead.       Blunt. No comfort.                        │
│   Connects your current       Asks: what is ACTUALLY                    │
│   problem to your long-       true here right now,                      │
│   term goals and direction.   not what you wish was                     │
│   Asks: does this move        true. Points to real                      │
│   you forward or sideways?    constraints and risks.                    │
│                                                                         │
│   Searches memory for:        Searches memory for:                      │
│   your goals, ambitions,      times you overcommitted,                  │
│   career direction,           got burned out, made                      │
│   values you stated           unrealistic plans,                        │
│                               misjudged situations                      │
│                                                                         │
│   THE ADVOCATE                THE CONTRARIAN                            │
│   ─────────────────           ─────────────────                         │
│   Fully on your side.         Friendly but stubborn.                    │
│   Asks: what do YOU           Specifically designed                     │
│   actually want — not         to push back on                           │
│   what you think you          whatever conclusion                       │
│   should want, not what       you seem to be leaning                    │
│   others expect. Protects     toward. Finds the                         │
│   your emotional needs.       uncomfortable angle                       │
│                               nobody else says.                         │
│   Searches memory for:        Searches memory for:                      │
│   your desires, emotional     decisions you later                       │
│   patterns, what makes        regretted, blind spots,                   │
│   you feel fulfilled          repeated mistakes,                        │
│   vs. drained                 patterns in your thinking                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Full Architecture — How It All Flows

```
USER TYPES A MESSAGE
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STM LOADER                                                           │
│  Loads current conversation history from LangGraph checkpoint         │
│  Everything said in THIS session — names, context, earlier turns      │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  PROBLEM CLASSIFIER  (Conditional Workflow)                           │
│                                                                       │
│  What kind of problem is this?                                        │
│                                                                       │
│  Career decision  →  Strategist gets extra weight                     │
│  Emotional issue  →  Advocate gets extra weight                       │
│  Practical plan   →  Realist gets extra weight                        │
│  Repeated pattern →  Contrarian gets extra weight                     │
│                                                                       │
│  Also decides: how many LTM memories to pull (2 or 5)                │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LTM FETCHER  (Runs before agents, feeds them all)                    │
│                                                                       │
│  Hits PostgreSQL with 4 different semantic search queries             │
│  simultaneously — one tailored to each advisor's perspective          │
│                                                                       │
│  Query A →  "user long term goals career direction ambition"          │
│  Query B →  "user past mistakes overwhelm unrealistic expectations"   │
│  Query C →  "user values desires emotional needs fulfillment"         │
│  Query D →  "user regrets blind spots repeated patterns mistakes"     │
│                                                                       │
│  Each query returns the top 3 most semantically relevant memories     │
│  Each advisor gets a DIFFERENT slice of the same memory store         │
└───────────────────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────┐                       ┌──────────────────┐
│  STRATEGIST      │                       │  REALIST         │
│  AGENT           │                       │  AGENT           │
│                  │                       │                  │
│  Gets:           │                       │  Gets:           │
│  - User message  │                       │  - User message  │
│  - STM context   │                       │  - STM context   │
│  - Memory set A  │                       │  - Memory set B  │
│                  │                       │                  │
│  Produces:       │                       │  Produces:       │
│  Pydantic object │                       │  Pydantic object │
│  {stance,        │                       │  {stance,        │
│   key_point,     │                       │   key_point,     │
│   memory_used,   │                       │   memory_used,   │
│   confidence}    │                       │   confidence}    │
└──────────────────┘                       └──────────────────┘
        │                                             │
        │          ALL FOUR RUN IN                    │
        │          PARALLEL  ──────────────────────── │
        │                                             │
┌──────────────────┐                       ┌──────────────────┐
│  ADVOCATE        │                       │  CONTRARIAN      │
│  AGENT           │                       │  AGENT           │
│                  │                       │                  │
│  Gets:           │                       │  Gets:           │
│  - User message  │                       │  - User message  │
│  - STM context   │                       │  - STM context   │
│  - Memory set C  │                       │  - Memory set D  │
│                  │                       │                  │
│  Produces:       │                       │  Produces:       │
│  Pydantic object │                       │  Pydantic object │
│  {stance,        │                       │  {stance,        │
│   key_point,     │                       │   key_point,     │
│   memory_used,   │                       │   memory_used,   │
│   confidence}    │                       │   confidence}    │
└──────────────────┘                       └──────────────────┘
        │                                             │
        └──────────────────┬──────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────────┐
│  SYNTHESIZER NODE  (Sequential — waits for all 4)                     │
│                                                                       │
│  Reads all four Pydantic objects                                      │
│  Finds: where do 3 or 4 agents agree? → that is signal               │
│  Finds: where is there strong disagreement? → that is tension         │
│  Picks the most memory-backed argument as most credible               │
│  Writes one final response that feels like a whole conversation       │
│  Not a summary — an actual recommendation with reasoning              │
└───────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────────┐
│  MEMORY WRITER  (Runs after response)                                 │
│                                                                       │
│  LLM reads the full conversation turn                                 │
│  Extracts: is there anything new worth remembering?                   │
│  Deduplication check: is this already in memory?                      │
│  If new → stores in PostgreSQL with embedding                         │
│  Examples of what gets stored:                                        │
│  "User is considering a team lead role at current company"            │
│  "User mentioned feeling undervalued at work"                         │
│  "User prioritized stability over growth in this session"             │
└───────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                     RESPONSE TO USER
```

---

## What Lives in PostgreSQL and Why

```
POSTGRESQL DATABASE
─────────────────────────────────────────────────────────────────────────

TABLE: memories
┌──────────┬─────────────────────────────────────────────┬──────────────────────┐
│  key     │  value (text)                               │  embedding (vector)  │
├──────────┼─────────────────────────────────────────────┼──────────────────────┤
│  mem_001 │  User's name is Aryan                       │  [0.23, -0.11, ...]  │
│  mem_002 │  User wants to become freelance consultant  │  [0.67,  0.34, ...]  │
│          │  within 2 years                             │                      │
│  mem_003 │  User feels drained by meetings and prefers │  [0.41, -0.29, ...]  │
│          │  deep focused work                          │                      │
│  mem_004 │  User turned down a promotion last year     │  [0.55,  0.18, ...]  │
│          │  and mentioned regretting it 3 months later │                      │
│  mem_005 │  User tends to underestimate how long tasks │  [-0.12, 0.88, ...]  │
│          │  take — mentioned this twice                │                      │
│  mem_006 │  User values family time above salary       │  [0.33,  0.44, ...]  │
│  mem_007 │  User is currently leading a Python project │  [0.71, -0.05, ...]  │
│          │  and leading a Python project               │                      │
│  mem_008 │  User feels most energized when solving     │  [0.29,  0.62, ...]  │
│          │  technical problems independently           │                      │
└──────────┴─────────────────────────────────────────────┴──────────────────────┘

Everything in the value column is plain readable text.
Everything in the embedding column is a list of 768 or 1536 numbers
representing the MEANING of that text in mathematical space.
```

---

## How Embeddings Actually Help Here

Think of it this way. You have 50 memories stored about a user. When the Strategist needs relevant context, you cannot just fetch all 50 — that would flood the LLM with noise. You need the 3 most relevant ones.

```
WITHOUT EMBEDDINGS  (keyword search)
──────────────────────────────────────────────────────────────────────
Query: "user career goals"

Memory "User wants to become freelance consultant"   → FOUND (has "career")
Memory "User values family time above salary"        → MISSED (no keyword match)
Memory "User feels energized solving problems alone" → MISSED (no keyword match)

You miss memories that are RELEVANT but don't share exact words.


WITH EMBEDDINGS  (semantic search)
──────────────────────────────────────────────────────────────────────
Query: "user long term career direction and ambition"
→ This query gets converted into a vector  [0.61, 0.29, -0.14, ...]

Now compare this vector against every stored memory vector using
cosine similarity  (how close are these two vectors in meaning space?)

Memory "User wants to become freelance consultant"    → similarity: 0.91  ✅
Memory "User values family time above salary"         → similarity: 0.78  ✅
Memory "User feels energized solving problems alone"  → similarity: 0.74  ✅
Memory "User's name is Aryan"                         → similarity: 0.12  ✗

Top 3 returned.  All three are genuinely useful for the Strategist.
None of them required an exact keyword match.

This is the RAG layer.  PostgreSQL stores the text.
The embedding index makes it semantically searchable.
```

---

## LTM vs STM — The Clear Distinction

```
SHORT TERM MEMORY (STM)
────────────────────────────────────────────────────────────────────────
What:    Everything said in the CURRENT conversation session
Where:   LangGraph's checkpoint system (also in Postgres but different table)
How long: Lives for the duration of one session / thread
Example: "Earlier in this conversation you said you were nervous about it"
Use:     Agents read STM to understand context of the current exchange


LONG TERM MEMORY (LTM)
────────────────────────────────────────────────────────────────────────
What:    Distilled facts extracted from ALL past sessions
Where:   PostgreSQL memories table with embedding index
How long: Forever — survives restarts, days, weeks
Example: "Three weeks ago you mentioned wanting to go freelance"
Use:     Agents do semantic search to pull relevant facts before speaking

Together:
  STM tells agents WHAT is happening right now in this conversation
  LTM tells agents WHO this person is based on everything they have shared
```

---

## Full End-to-End Example

**User types:**
> *"My manager just offered me a team lead position. More money, more responsibility. I'm not sure if I want it. What do I do?"*

---

**Step 1 — Problem Classifier**

Reads the message. Detects: career decision with emotional uncertainty. Labels it `CAREER_DECISION`. Sets advisor weights: Strategist and Advocate get slightly more space. Decides to fetch 3 memories per advisor.

---

**Step 2 — LTM Fetcher runs 4 semantic searches simultaneously**

```
Query for Strategist: "career goals long term direction ambition future plans"
→ Returns:
   mem_002: "User wants to become freelance consultant within 2 years"
   mem_008: "User feels energized solving technical problems independently"
   mem_007: "User is currently leading a Python project at work"

Query for Realist: "past mistakes overcommitment stress burnout workload problems"
→ Returns:
   mem_005: "User tends to underestimate how long tasks take — mentioned twice"
   mem_003: "User feels drained by meetings and prefers deep focused work"
   mem_007: "User is currently leading a Python project at work"

Query for Advocate: "values desires emotional needs what makes user happy fulfilled"
→ Returns:
   mem_006: "User values family time above salary"
   mem_008: "User feels energized solving technical problems independently"
   mem_003: "User feels drained by meetings and prefers deep focused work"

Query for Contrarian: "regrets blind spots decisions that went wrong repeated patterns"
→ Returns:
   mem_004: "User turned down a promotion last year and regretted it 3 months later"
   mem_005: "User tends to underestimate how long tasks take — mentioned twice"
   mem_002: "User wants to become freelance consultant within 2 years"
```

Notice: all four got different memory sets from the same store. Same data, four different lenses.

---

**Step 3 — All Four Agents Speak (in parallel)**

```
┌─────────────────────────────────────────────────────────────────────┐
│  STRATEGIST                                                         │
│                                                                     │
│  Memory used: "wants freelance in 2 years" + "energized alone"     │
│                                                                     │
│  "Team lead moves you further from the freelance path you           │
│   described. Leading a team means your time becomes about           │
│   managing people, not building things. The question is whether     │
│   this role teaches skills you need for consulting — like client    │
│   management, scoping projects, handling expectations — or          │
│   whether it just ties you deeper into this company."               │
│                                                                     │
│  Confidence: 0.84   Memory referenced: mem_002, mem_008            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  REALIST                                                            │
│                                                                     │
│  Memory used: "drained by meetings" + "underestimates time"        │
│                                                                     │
│  "You already feel drained by meetings in your current role.        │
│   A team lead position is largely meetings, check-ins, performance  │
│   reviews, unblocking people. You have also mentioned twice that    │
│   you underestimate how long things take. Adding management         │
│   overhead to an already stretched capacity is a concrete risk,     │
│   not a hypothetical one."                                          │
│                                                                     │
│  Confidence: 0.91   Memory referenced: mem_003, mem_005            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  ADVOCATE                                                           │
│                                                                     │
│  Memory used: "values family time" + "energized by solo work"      │
│                                                                     │
│  "What you have said you actually want is time with family and      │
│   the feeling you get when you are deep in a technical problem      │
│   by yourself. Team lead takes both of those. More hours, less      │
│   deep work. The money is real but ask yourself honestly: will      │
│   you resent this role in 6 months when you are in your fourth      │
│   meeting of the day and your code is sitting untouched?"           │
│                                                                     │
│  Confidence: 0.88   Memory referenced: mem_006, mem_008            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONTRARIAN                                                         │
│                                                                     │
│  Memory used: "turned down promotion, regretted it"                │
│                                                                     │
│  "You turned down a promotion last year. You regretted it.          │
│   Now you are hesitating again. I am not saying take this one —     │
│   the situations may be different. But notice the pattern.          │
│   You tend to frame growth opportunities as threats and then        │
│   feel the cost of that choice later. What specifically is          │
│   different about this offer compared to the last one?"             │
│                                                                     │
│  Confidence: 0.87   Memory referenced: mem_004, mem_002            │
└─────────────────────────────────────────────────────────────────────┘
```

---

**Step 4 — Synthesizer reads all four and responds**

```
WHERE THEY AGREE (3 or 4 agents pointing the same way)
───────────────────────────────────────────────────────
Strategist + Realist + Advocate all say: this role conflicts
with who you actually are and what you said you want.
That is strong signal. Three independent lenses, same conclusion.

WHERE THEY DISAGREE (tension worth naming)
───────────────────────────────────────────────────────
Contrarian is the outlier: raises the pattern of turning
things down and regretting it. This does not say take the
job — it says examine WHY you are hesitating before deciding.

MOST MEMORY-BACKED ARGUMENT
───────────────────────────────────────────────────────
Realist has the highest confidence and cites two separate
past memories. Weight it accordingly.
```

**Final response to user:**

> Three of your four advisors are pointing the same direction: this role as described conflicts with what you have said you actually want — deep technical work, less meeting overhead, and a path toward independence. The Realist is blunt about it: you have said twice that meetings drain you and you already stretch your capacity. Adding team management to that is a predictable problem, not a risk.
>
> The one tension worth sitting with comes from your history. You turned down something similar last year and felt the cost of that later. That does not mean say yes — it means ask yourself one honest question before deciding: am I hesitating because this genuinely does not fit my life, or am I hesitating because growth feels uncomfortable and I am pattern-matching to safety?
>
> If the answer is the first one, turn it down clearly. If the answer is the second one, that is worth a longer conversation.
>
> What does your gut say when you imagine yourself 6 months into this role?

---

**Step 5 — Memory Writer**

Reads the full exchange. Extracts new facts worth keeping:

```
NEW: "User was offered a team lead role at current company"      → stored
NEW: "User expressed uncertainty about whether management        → stored
      aligns with their goals"
DUPLICATE: "User values deep technical work"                    → skipped
           (already in mem_008, deduplication catches it)
```

---

## The Whole Thing in One Line

Every time you talk to it, four agents who know your history look at your problem from different angles at the same time, a synthesizer finds the signal in what they all said, and everything new you revealed gets quietly remembered for next time. The longer you use it the more accurate it gets — because the memory grows while the architecture stays the same.
