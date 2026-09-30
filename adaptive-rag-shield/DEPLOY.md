# Adaptive RAG Shield -- local-Llama deployment

## 1. Get model access
`meta-llama/Meta-Llama-3-8B-Instruct` is gated: accept the license at
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct , then create a read token at
https://huggingface.co/settings/tokens . If you'd rather skip the approval wait, use the ungated
mirror `NousResearch/Meta-Llama-3-8B-Instruct` (same weights) by setting `HF_MODEL` below.

## 2. Choose a GPU host
Llama-3-8B-Instruct needs roughly:
  - 4-bit (default, `HF_LOAD_4BIT=1`): ~6-8GB VRAM  -> a free Colab T4 (16GB) works
  - fp16  (`HF_LOAD_4BIT=0`):          ~16GB VRAM  -> needs a bigger card (A10/A100/L4)

For a persistent public service (not Colab), rent one GPU instance on **RunPod, Lambda Labs, or
a GCP/AWS GPU VM** with an NVIDIA driver + the NVIDIA container toolkit installed. A single T4 or
L4 instance is enough for 4-bit inference; scale to more replicas (each with its own GPU) if you
need to serve more concurrent users, since this framework loads one model per process.

## 3. Run it

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx          # from step 1 (skip if using the ungated NousResearch mirror)
export API_KEYS=some-long-random-key,another-key-per-team

docker compose up --build
```

First start downloads the model (~16GB) into the `hf-cache` volume -- expect several minutes.
Subsequent restarts reuse the cached weights.

Check it's alive:
```bash
curl http://localhost:8000/health
```

Call it:
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Authorization: Bearer some-long-random-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I return an item?"}'
```

## 4. Config knobs (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `HF_MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | swap for `NousResearch/Meta-Llama-3-8B-Instruct` (ungated) or `mistralai/Mistral-7B-Instruct-v0.3` |
| `HF_LOAD_4BIT` | `1` | set `0` only if you have >=20GB VRAM and want full-precision quality |
| `HF_MAX_NEW_TOKENS` | `256` | cap on generated reply length (latency/cost knob) |
| `HF_TOKEN` | -- | required for gated models; never bake into the image, pass at run time |
| `API_KEYS` | `dev-key-change-me` | comma-separated; **change before exposing publicly** |
| `RATE_LIMIT_PER_MIN` | `30` | per-key requests/minute |

## 5. Known limitations of this setup

- **Single process = single model copy.** `--workers 1` in the Dockerfile is intentional: a second
  worker would load a second full copy of Llama onto the GPU and likely OOM. To serve more traffic,
  run more container replicas (each on its own GPU) behind a load balancer, not more workers on one GPU.
- **In-memory rate limiter and audit log** reset if the container restarts and don't sync across
  replicas. Fine for a single instance; swap for Redis (rate limits/sessions) and Postgres (audit
  log) before running multiple replicas -- see the note in the main deployment writeup.
- **Cold start is slow.** Model loading happens at process startup (blocks `/health` until done).
  Give your hosting platform's health check a generous startup grace period (2-3 minutes) so it
  doesn't kill the container before the model finishes loading.
- I could not actually run this end-to-end here (no GPU or internet in this sandbox) -- I verified
  the code's control flow against mocked `torch`/`transformers` and confirmed the Dockerfile/compose
  file structure, but you should smoke-test on your own GPU before trusting it in production.
