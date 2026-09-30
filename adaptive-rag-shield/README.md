# Adaptive RAG Shield

A multi-layer prompt-injection defense framework for RAG-based chatbots, combining ideas from:
- **PromptShield** (low-FPR threshold calibration)
- **DataSentinel** (known-answer detection, pluggable)
- **PromptGuard** (regex+classifier input screening, LLM-as-critic, Adaptive Response Refinement)
- **"A Layered Security Framework Against Prompt Injection in RAG-Based Chatbots"** (L1/L2/L3 layered
  architecture, provenance-tagged context assembly, continuous audit loop with alerting/retraining)

## What's here

| File | Purpose |
|---|---|
| `adaptive_rag_shield.py` | The framework: L1 input screening, L2 privilege-constrained context assembly, L3 output auditing, ARR, audit loop, ablation harness, synthetic benchmark |
| `service/main.py` | FastAPI wrapper: `/v1/chat`, API-key auth, rate limiting, `/v1/feedback` + `/v1/admin/retrain` for the audit loop |
| `service/Dockerfile` | GPU image (CUDA) for a local Llama-3/Mistral backend via `transformers` + `bitsandbytes` |
| `service/Dockerfile.free` | CPU-only image for free hosting (Hugging Face Spaces) via a quantized GGUF model + `llama-cpp-python` |
| `docker-compose.yml` | Local/VM run with GPU reservation (works on a raw GPU VM — Lambda Labs, GCP, AWS; **not** inside a RunPod Pod, see DEPLOY.md) |
| `DEPLOY.md` | Full deployment walkthrough: RunPod, Lambda Labs, and free (Colab / Hugging Face Spaces) paths |

## Quick start (no GPU needed — mock LLM, for testing the defense logic itself)

```bash
pip install scikit-learn numpy fastapi "uvicorn[standard]" pydantic
python adaptive_rag_shield.py demo        # a few example queries incl. obfuscated + poisoned-doc cases
python adaptive_rag_shield.py ablation    # ASR/FPR across all 8 layer-subset configurations
```

## Run the API

```bash
cd service
export LLM_BACKEND=mock API_KEYS=dev-key-change-me   # or hf / llamacpp / openai — see DEPLOY.md
uvicorn main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"message": "How do I return an item?"}'
```

## Deploying for other people to use

See **[DEPLOY.md](DEPLOY.md)** for:
- Paid, real-time: RunPod (GPU pods, no Docker needed) or Lambda Labs (`docker compose up` as-is)
- Free: Google Colab + tunnel (temporary, GPU) or Hugging Face Spaces (persistent, CPU-only, slower)

## Known limitations

- The ablation numbers in this repo were produced against a **simulated** vulnerable LLM and a small
  synthetic dataset — they demonstrate the layering logic, not real-world attack success rates.
- DataSentinel's adversarial minimax *training* of the detector (GCG-based) is not implemented here;
  `KnownAnswerDetector` accepts any detection LLM via `llm_fn`, so a separately fine-tuned model can
  be plugged in.
- Not yet validated against real Llama-3 output — do this before trusting it in production (see
  the "what to check before deploying" note in DEPLOY.md).
