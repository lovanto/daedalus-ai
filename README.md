# Daedalus AI Layer

Python 3.11 · FastAPI · Ollama — LLM-powered assist endpoints for the ADLC platform.

This service is **internal-only**. The frontend never calls it directly; the Go API proxies all requests.

## Overview

The AI layer wraps a locally running Ollama instance (or Anthropic cloud fallback) and exposes structured endpoints for each ADLC phase:

| Router | Prefix | Purpose |
|--------|--------|---------|
| `assist` | `/ai/assist/` | Define suggestions, system-prompt generation, tune fixes |
| `eval` | `/ai/eval/` | Suggest eval cases, classify failures, run & score cases |
| `analyze` | `/ai/analyze/` | Observation pattern analysis, scope-drift detection |

## Tech stack

| Dependency | Version |
|---|---|
| Python | 3.11 |
| FastAPI | 0.115.5 |
| Uvicorn | 0.32.1 |
| httpx | 0.27.2 |
| asyncpg | 0.30.0 |
| Pydantic | 2.10.3 |
| pydantic-settings | 2.6.1 |

## Project layout

```
backend/ai/
├── main.py              # FastAPI app, middleware, lifecycle hooks
├── config.py            # Settings via pydantic-settings / .env
├── db.py                # asyncpg connection pool
├── utils.py             # extract_json helper
├── ollama/
│   ├── client.py        # OllamaClient — generate(), is_reachable()
│   └── prompts.py       # All prompt-builder functions
├── models/
│   └── schemas.py       # Pydantic request/response models
└── routers/
    ├── assist.py        # /ai/assist/* endpoints
    ├── eval.py          # /ai/eval/* endpoints
    └── analyze.py       # /ai/analyze/* endpoints
```

## Configuration

Copy `.env.example` to `.env` (or set environment variables directly):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=120.0
OLLAMA_MAX_RETRIES=3

# Set to "cloud" to use Anthropic instead of local Ollama
OLLAMA_MODE=local
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-6

DATABASE_URL=postgres://daedalus:daedalus@localhost:5432/daedalus
PYTHON_AI_PORT=8001
GO_API_URL=http://localhost:3010
```

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the service
uvicorn main:app --reload --port 8001
```

On startup the service pings Ollama and logs which models are available. If the configured model is not pulled locally you will see a warning — run `ollama pull <model>` to fix it.

## API endpoints

### Health

```
GET /ai/health
```

Returns service status and Ollama reachability.

### Assist

```
POST /ai/assist/define
POST /ai/assist/build/system-prompt
POST /ai/assist/tune/suggest-fix
```

### Eval

```
POST /ai/eval/suggest-cases
POST /ai/eval/classify-failure
POST /ai/eval/run-case
```

### Analyze

```
POST /ai/analyze/patterns
POST /ai/analyze/scope-drift
```

Interactive docs are available at `http://localhost:8001/docs` (Swagger UI) and `http://localhost:8001/redoc`.

## Docker

```bash
docker build -t daedalus-ai .
docker run -p 8001:8001 --env-file .env daedalus-ai
```

Or via the root `docker-compose.yml`:

```bash
docker-compose up ai
```
