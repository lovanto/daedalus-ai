from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse
from ollama.client import OllamaClient, OllamaError, OllamaTimeoutError

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# POST /ai/chat
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Hold a live multi-turn conversation with an agent using its build config",
)
async def chat(req: ChatRequest) -> ChatResponse:
    """Run a turn of the agent under test.

    The agent's persona is its latest build's system prompt (injected by the Go
    proxy). The whole conversation is replayed each turn so the model has the
    full context — this endpoint is stateless, the FE owns the transcript."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    client = OllamaClient()
    try:
        reply = await client.chat(
            messages,
            model=req.model or None,
            system=req.system_prompt or None,
        )
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return ChatResponse(reply=reply.strip())
