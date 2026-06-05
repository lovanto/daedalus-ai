from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from models.schemas import (
    AssistDefineRequest,
    AssistDefineResponse,
    BuildSystemPromptRequest,
    BuildSystemPromptResponse,
    DefineSuggestion,
    TuneFixRequest,
    TuneFixResponse,
    TuneFixSuggestion,
)
from ollama.client import OllamaClient, OllamaError, OllamaTimeoutError
from ollama.prompts import build_system_prompt, define_assist, suggest_tune_fix
from utils import extract_json

router = APIRouter(prefix="/assist", tags=["assist"])


async def _ollama_generate(prompt: str, temperature: float = 0.7) -> str:
    client = OllamaClient()
    try:
        return await client.generate(prompt, temperature=temperature)
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /ai/assist/define
# ---------------------------------------------------------------------------

@router.post("/define", response_model=AssistDefineResponse, summary="AI suggestions for a Define section")
async def assist_define(req: AssistDefineRequest) -> AssistDefineResponse:
    prompt = define_assist(
        section=req.section,
        agent_name=req.agent_name,
        agent_description=req.agent_description,
        existing_content=req.existing_content,
    )

    raw = await _ollama_generate(prompt, temperature=0.8)
    parsed, had_error = extract_json(raw)

    if had_error or not isinstance(parsed, dict):
        # Graceful fallback: wrap raw text as a single suggestion
        return AssistDefineResponse(
            suggestions=[
                DefineSuggestion(
                    title="AI suggestion",
                    content=raw,
                    reasoning="Raw response — JSON parsing failed",
                )
            ]
        )

    suggestions = []
    for item in parsed.get("suggestions", []):
        if isinstance(item, dict):
            suggestions.append(
                DefineSuggestion(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    reasoning=item.get("reasoning", ""),
                )
            )

    if not suggestions:
        suggestions = [
            DefineSuggestion(title="AI suggestion", content=raw, reasoning="")
        ]

    return AssistDefineResponse(suggestions=suggestions)


# ---------------------------------------------------------------------------
# POST /ai/assist/build/system-prompt
# ---------------------------------------------------------------------------

@router.post(
    "/build/system-prompt",
    response_model=BuildSystemPromptResponse,
    summary="Generate a system prompt from an agent definition",
)
async def assist_system_prompt(req: BuildSystemPromptRequest) -> BuildSystemPromptResponse:
    prompt = build_system_prompt(req.agent_definition.model_dump())
    system_prompt_text = await _ollama_generate(prompt, temperature=0.6)
    # Strip any accidental markdown fences the model may have added
    system_prompt_text = re.sub(r"^```[^\n]*\n?", "", system_prompt_text.strip())
    system_prompt_text = re.sub(r"\n?```$", "", system_prompt_text.strip())
    return BuildSystemPromptResponse(system_prompt=system_prompt_text.strip())


# ---------------------------------------------------------------------------
# POST /ai/assist/tune/suggest-fix
# ---------------------------------------------------------------------------

@router.post(
    "/tune/suggest-fix",
    response_model=TuneFixResponse,
    summary="Suggest specific fixes for a tune cycle",
)
async def assist_tune_fix(req: TuneFixRequest) -> TuneFixResponse:
    prompt = suggest_tune_fix(
        failure_type=req.failure_type,
        failure_description=req.failure_description,
        current_build=req.current_build,
    )

    raw = await _ollama_generate(prompt, temperature=0.7)
    parsed, had_error = extract_json(raw)

    if had_error or not isinstance(parsed, dict):
        return TuneFixResponse(
            suggestions=[
                TuneFixSuggestion(
                    change_type="general",
                    description=raw,
                    expected_impact="See raw AI response above",
                )
            ]
        )

    suggestions = []
    for item in parsed.get("suggestions", []):
        if isinstance(item, dict):
            suggestions.append(
                TuneFixSuggestion(
                    change_type=item.get("change_type", "general"),
                    description=item.get("description", ""),
                    expected_impact=item.get("expected_impact", ""),
                )
            )

    if not suggestions:
        suggestions = [
            TuneFixSuggestion(
                change_type="general",
                description=raw,
                expected_impact="",
            )
        ]

    return TuneFixResponse(suggestions=suggestions)
