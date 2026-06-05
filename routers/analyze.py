from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import (
    AnalyzePatternsRequest,
    AnalyzePatternsResponse,
    ScopeDriftRequest,
    ScopeDriftResponse,
)
from ollama.client import OllamaClient, OllamaError, OllamaTimeoutError
from ollama.prompts import analyze_patterns, check_scope_drift
from utils import extract_json

router = APIRouter(prefix="/analyze", tags=["analyze"])


async def _generate(prompt: str, temperature: float = 0.4) -> str:
    client = OllamaClient()
    try:
        return await client.generate(prompt, temperature=temperature)
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /ai/analyze/patterns
# ---------------------------------------------------------------------------

@router.post(
    "/patterns",
    response_model=AnalyzePatternsResponse,
    summary="Analyze observation patterns and recommend a routing action",
)
async def analyze_patterns_endpoint(req: AnalyzePatternsRequest) -> AnalyzePatternsResponse:
    if not req.observations:
        return AnalyzePatternsResponse(
            key_patterns=[],
            severity_summary="No observations provided.",
            recommended_route="monitor",
        )

    observations_text = "\n".join(f"- {obs}" for obs in req.observations)
    prompt = analyze_patterns(observations_text=observations_text, agent_name=req.agent_name)

    raw = await _generate(prompt, temperature=0.5)
    parsed, had_error = extract_json(raw)

    if had_error or not isinstance(parsed, dict):
        return AnalyzePatternsResponse(
            key_patterns=[raw],
            severity_summary="JSON parsing failed — see key_patterns for raw response.",
            recommended_route="monitor",
        )

    return AnalyzePatternsResponse(
        key_patterns=parsed.get("key_patterns", []),
        severity_summary=parsed.get("severity_summary", ""),
        recommended_route=parsed.get("recommended_route", "monitor"),
        recommended_route_reasoning=parsed.get("recommended_route_reasoning", ""),
    )


# ---------------------------------------------------------------------------
# POST /ai/analyze/scope-drift
# ---------------------------------------------------------------------------

@router.post(
    "/scope-drift",
    response_model=ScopeDriftResponse,
    summary="Detect scope drift by comparing observations against the agent definition",
)
async def scope_drift_endpoint(req: ScopeDriftRequest) -> ScopeDriftResponse:
    if not req.observations:
        return ScopeDriftResponse(
            drift_detected=False,
            drift_areas=[],
            drift_severity="low",
            recommendation="No observations to analyze.",
        )

    observations_text = "\n".join(f"- {obs}" for obs in req.observations)
    prompt = check_scope_drift(
        observations_text=observations_text,
        agent_definition=req.agent_definition.model_dump(),
    )

    raw = await _generate(prompt, temperature=0.3)
    parsed, had_error = extract_json(raw)

    if had_error or not isinstance(parsed, dict):
        return ScopeDriftResponse(
            drift_detected=False,
            drift_areas=[],
            drift_severity="unknown",
            recommendation=raw,
        )

    return ScopeDriftResponse(
        drift_detected=bool(parsed.get("drift_detected", False)),
        drift_areas=parsed.get("drift_areas", []),
        drift_severity=parsed.get("drift_severity", "low"),
        recommendation=parsed.get("recommendation", ""),
    )
