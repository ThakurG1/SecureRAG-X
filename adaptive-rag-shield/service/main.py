"""
main.py — FastAPI wrapper around RAGShield so other users can call the framework over HTTP.

Run locally:
    uvicorn main:app --reload --port 8000

Auth: every request needs header  Authorization: Bearer <API_KEY>
Keys are read from the API_KEYS env var (comma-separated) or api_keys.txt (one per line).
Each key gets its own conversation sessions and its own rate limit bucket.
"""
from __future__ import annotations

import os, sys, time, uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_rag_shield import (
    Config, KB, RAGShield, SYSTEM_PROMPT, TfidfRetriever, build_pipeline, build_dataset,
    hf_llm, llama_cpp_llm, openai_llm, make_mock_llm,
)

# ----------------------------------------------------------------------------------
# Config from environment
# ----------------------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf")             # mock | openai | hf | llamacpp
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_LOAD_4BIT = os.getenv("HF_LOAD_4BIT", "1") == "1"      # 4-bit needed on <16GB VRAM (e.g. free T4)
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
GGUF_REPO = os.getenv("GGUF_REPO", "bartowski/Meta-Llama-3-8B-Instruct-GGUF")   # llamacpp backend (free, CPU-only)
GGUF_FILE = os.getenv("GGUF_FILE", "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", str(60 * 60)))  # evict idle sessions after 1h


def load_api_keys() -> set[str]:
    keys = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
    f = Path(__file__).parent / "api_keys.txt"
    if f.exists():
        keys |= {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
    if not keys:
        keys = {"dev-key-change-me"}   # local-dev fallback; ALWAYS override in production
    return keys


API_KEYS = load_api_keys()

# ----------------------------------------------------------------------------------
# Build the shield once at startup (shared across all users; sessions are per-key)
# ----------------------------------------------------------------------------------
print(f"[startup] LLM_BACKEND={LLM_BACKEND}", flush=True)
if LLM_BACKEND == "openai":
    llm = openai_llm(os.getenv("OPENAI_MODEL", "gpt-4o"))
elif LLM_BACKEND == "hf":
    print(f"[startup] loading {HF_MODEL} (4bit={HF_LOAD_4BIT}) — this can take a minute or two...", flush=True)
    llm = hf_llm(HF_MODEL, load_4bit=HF_LOAD_4BIT, max_new_tokens=HF_MAX_NEW_TOKENS)
    print("[startup] model loaded.", flush=True)
elif LLM_BACKEND == "llamacpp":
    print(f"[startup] downloading/loading GGUF {GGUF_REPO}/{GGUF_FILE} (CPU) — first run can take a while...", flush=True)
    llm = llama_cpp_llm(GGUF_REPO, GGUF_FILE, max_new_tokens=HF_MAX_NEW_TOKENS)
    print("[startup] model loaded.", flush=True)
else:
    llm = make_mock_llm()

_data = build_dataset(seed=0)  # trains the L1 classifier & calibrates thresholds/drift once
_shield: RAGShield = build_pipeline(_data, Config())
_shield.llm = llm                      # swap in the real backend chosen above
_shield.retriever = TfidfRetriever(KB)  # replace with your own knowledge base retriever

# ----------------------------------------------------------------------------------
# Simple in-memory rate limiter + session registry (swap for Redis at scale — see README)
# ----------------------------------------------------------------------------------
_buckets: dict[str, deque] = defaultdict(deque)
_last_seen: dict[str, float] = {}


def check_rate_limit(key: str):
    now = time.time()
    q = _buckets[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Rate limit exceeded, try again shortly.")
    q.append(now)


def auth(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header.")
    key = authorization.removeprefix("Bearer ").strip()
    if key not in API_KEYS:
        raise HTTPException(401, "Invalid API key.")
    return key


# ----------------------------------------------------------------------------------
# API models
# ----------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: Optional[str] = None
    docs: Optional[list[str]] = None   # override retrieval with explicit documents (optional)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    status: str                # delivered | blocked_l1 | blocked_l3 | escalated
    layers_triggered: list[str]
    latency_ms: float


class FeedbackRequest(BaseModel):
    text: str
    is_attack: bool


app = FastAPI(title="Adaptive RAG Shield API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "backend": LLM_BACKEND, "sessions_active": len(_last_seen)}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, authorization: Optional[str] = Header(None)):
    api_key = auth(authorization)
    check_rate_limit(api_key)
    session_id = req.session_id or f"{api_key[:8]}-{uuid.uuid4().hex[:8]}"
    _last_seen[session_id] = time.time()
    result = _shield.run(req.message, session_id=session_id, retrieved=req.docs)
    return ChatResponse(session_id=session_id, response=result.response, status=result.status,
                        layers_triggered=result.layers, latency_ms=round(result.latency_ms, 2))


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest, authorization: Optional[str] = Header(None)):
    """Lets a reviewer confirm a true/false positive; folds it into the audit loop's retraining queue."""
    auth(authorization)
    _shield.audit.add_feedback(req.text, req.is_attack)
    return {"queued": True, "pending_feedback": len(_shield.audit.feedback)}


@app.post("/v1/admin/retrain")
def retrain(authorization: Optional[str] = Header(None)):
    auth(authorization)
    n = _shield.audit.retrain(_shield.screen.clf)
    return {"retrained_on": n}


@app.get("/v1/admin/alerts")
def alerts(authorization: Optional[str] = Header(None)):
    auth(authorization)
    fired, z = _shield.audit.population_alert()
    stale = [s for s, t in list(_last_seen.items()) if time.time() - t > SESSION_TTL_S]
    for s in stale:
        _last_seen.pop(s, None)
    return {"population_alert": fired, "z_score": round(z, 3),
            "sessions_tracked": len(_last_seen), "events_logged": len(_shield.audit.events)}
