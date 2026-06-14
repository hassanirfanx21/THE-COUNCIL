# The Council — Full Deployment Guide

## What Was Built

| File | Purpose |
|---|---|
| `frontend/` | React + Vite app — dark purple neon UI |
| `backend/main.py` | FastAPI server wrapping the LangGraph workflow |
| `backend/council_graph.py` | Full LangGraph graph (extracted from notebook) |
| `backend/Dockerfile` | Docker build for Hugging Face Spaces |
| `frontend/vercel.json` | Vercel proxy config |

---

## Stack

| Piece | Where | Cost |
|---|---|---|
| Postgres + pgvector | **Supabase** | Free |
| FastAPI backend | **HuggingFace Spaces** | Free |
| React frontend | **Vercel** | Free |
| LLM | **Groq** | Free tier |

---

## Step 1 — Supabase (Postgres)

1. Go to [supabase.com](https://supabase.com) → **New Project**
2. **Settings → Database → URI** — copy the connection string:
   ```
   postgresql://postgres:[PASSWORD]@db.XXXX.supabase.co:5432/postgres
   ```
3. **Database → Extensions** → search `vector` → **Enable**
4. That's it. Your DB is ready.

> [!IMPORTANT]
> Supabase requires SSL. Your `DB_URI` env var must end with `?sslmode=require`.
> Set `SSL_MODE=require` env var in HF Spaces.

---

## Step 2 — HuggingFace Spaces (Backend)

### Create the Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **New Space**
2. **SDK: Docker** | give it a name e.g. `the-council-api`
3. **Visibility: Public** (or Private — both work)

### Push your backend

```bash
# In your terminal (The Council\backend folder)
git init
git add .
git commit -m "initial"

# Add HF remote (replace YOUR_USERNAME and SPACE_NAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
git push hf main
```

> [!NOTE]
> HF Spaces clones via HTTPS. You'll need to log in with `huggingface-cli login` 
> or use your HF token as the git password.

### Set Environment Variables in HF Spaces

Go to your Space → **Settings → Variables and Secrets**:

| Variable | Value |
|---|---|
| `DB_URI` | `postgresql://postgres:[PW]@db.XXXX.supabase.co:5432/postgres` |
| `SSL_MODE` | `require` |
| `GROQ_API_KEY` | Your Groq key |

> [!WARNING]
> First build takes 5-10 minutes (downloads `all-mpnet-base-v2` ~400MB).
> Subsequent restarts are fast because Docker layer caching keeps the model.

### Your API URL will be:
```
https://YOUR_USERNAME-SPACE_NAME.hf.space
```

Test it:
```bash
curl https://YOUR_USERNAME-SPACE_NAME.hf.space/health
# → {"status":"ok"}
```

---

## Step 3 — Vercel (Frontend)

### Update vercel.json

Open `frontend/vercel.json` and replace the destination:
```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "https://YOUR_USERNAME-SPACE_NAME.hf.space/$1" }
  ]
}
```

### Deploy

```bash
# In The Council\frontend folder
npm install -g vercel   # if not already installed
vercel login
vercel --prod
```

Vercel will ask:
- **Set up and deploy?** → Y
- **Which scope?** → your account
- **Link to existing project?** → N
- **Project name?** → `the-council` (or anything)
- **Directory?** → `./` (current)
- **Override build settings?** → N

Your frontend will be live at `https://the-council-XXXX.vercel.app` 🎉

---

## Step 4 — Local Dev (before deploying)

```bash
# Start backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Start frontend (different terminal)
cd frontend
npm run dev
```

Frontend at `http://localhost:5173` → proxies `/api` → `http://localhost:8000`

> [!TIP]
> The frontend works in **demo mode** even without the backend — it uses a mock response 
> so you can test the UI. The mock activates automatically on API errors.

---

## Updating After Deployment

### Backend change:
```bash
cd backend
git add . && git commit -m "update"
git push hf main   # HF rebuilds automatically
```

### Frontend change:
```bash
cd frontend
vercel --prod   # redeploys in ~30 seconds
```

---

## Architecture

```
User → Vercel (React)
         ↓ /api/*
   HuggingFace Space (FastAPI)
         ↓
   ┌─────────────────────────┐
   │   LangGraph Workflow    │
   │  STM → Classify → LTM  │
   │  [4 agents in parallel] │
   │       Synthesizer       │
   └──────────┬──────────────┘
              ↓
         Supabase (Postgres + pgvector)
         ├── PostgresSaver (STM / checkpoints)
         └── PostgresStore (LTM / embeddings)
```

