from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException

from models.schemas import (
    ApplyBuildPlan,
    ApplyDefinitionPlan,
    ApplyPlanRequest,
    ApplyPlanResponse,
    ApplyTargetNote,
    AssistDefineRequest,
    AssistDefineResponse,
    BuildSystemPromptRequest,
    BuildSystemPromptResponse,
    DefineSuggestion,
    RewritePromptRequest,
    RewritePromptResponse,
    TuneFixRequest,
    TuneFixResponse,
    TuneFixSuggestion,
)
from ollama.client import OllamaClient, OllamaError, OllamaTimeoutError
from ollama.prompts import (
    build_system_prompt,
    define_assist,
    rewrite_system_prompt,
    suggest_tune_fix,
    tune_definition_plan,
)
from utils import coerce_suggestions, extract_json, extract_temperature, salvage_objects, strip_code_fences

router = APIRouter(prefix="/assist", tags=["assist"])


async def _ollama_generate(prompt: str, temperature: float = 0.7) -> str:
    client = OllamaClient()
    try:
        return await client.generate(prompt, temperature=temperature)
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


async def _ollama_generate_thinking(prompt: str, temperature: float = 0.7) -> tuple[str, str]:
    """Generate and also return the model's reasoning trace (answer, thinking)."""
    client = OllamaClient()
    try:
        return await client.generate_with_thinking(prompt, temperature=temperature)
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _unwrap_str(value, default: str = "") -> str:
    """
    Coerce a suggestion field to a clean string.

    Models sometimes nest a whole object/array where a plain string is expected
    (e.g. a ``description`` that is itself ``{"suggestions": [...]}``). Render
    those as compact JSON rather than letting them fall through as ``[object]``
    or crash Pydantic; pass real strings through untouched.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value or default
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


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

    # Lower temperature keeps the structured JSON output reliable. Thinking is
    # opt-in (req.think): it surfaces the reasoning trace but is much slower, so
    # batch drafts leave it off and the interactive assist turns it on.
    if req.think:
        raw, thinking = await _ollama_generate_thinking(prompt, temperature=0.4)
    else:
        raw = await _ollama_generate(prompt, temperature=0.4)
        thinking = ""
    parsed, _ = extract_json(raw)

    # Recover suggestions from the well-formed object as well as the malformed
    # array shapes (line-split JSON, bare string lists) the model sometimes emits.
    items = coerce_suggestions(parsed, raw, keys=("title", "content"), text_key="content")
    suggestions = [
        DefineSuggestion(
            title=item.get("title", "").strip() or "AI suggestion",
            content=item.get("content", ""),
            reasoning=item.get("reasoning", ""),
        )
        for item in items
        if isinstance(item, dict) and (item.get("content", "").strip() or item.get("title", "").strip())
    ]

    if not suggestions:
        # Nothing recoverable — surface a clean message, not a JSON dump.
        suggestions = [
            DefineSuggestion(
                title="AI suggestion",
                content="The AI response could not be parsed. Try again — the "
                "model may have returned malformed output.",
                reasoning="",
            )
        ]

    return AssistDefineResponse(suggestions=suggestions, thinking=thinking)


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
        eval_history=[e.model_dump() for e in req.eval_history],
        failing_cases=[c.model_dump() for c in req.failing_cases],
    )

    raw = await _ollama_generate(prompt, temperature=0.7)
    parsed, _ = extract_json(raw)

    # Recover suggestion dicts from the object shape, the malformed array shapes
    # (line-split JSON, bare lists), or — failing that — salvaged objects.
    items = coerce_suggestions(
        parsed, raw, keys=("change_type", "description", "expected_impact")
    )

    suggestions = [
        TuneFixSuggestion(
            change_type=_unwrap_str(item.get("change_type"), "general"),
            description=_unwrap_str(item.get("description"), ""),
            expected_impact=_unwrap_str(item.get("expected_impact"), ""),
        )
        for item in items
    ]

    if not suggestions:
        # Nothing recoverable — surface a clean message, not a JSON dump.
        suggestions = [
            TuneFixSuggestion(
                change_type="general",
                description="The AI response could not be parsed. Try again — "
                "the model may have returned incomplete output.",
                expected_impact="",
            )
        ]

    return TuneFixResponse(suggestions=suggestions)


# ---------------------------------------------------------------------------
# POST /ai/assist/tune/rewrite-prompt
# ---------------------------------------------------------------------------

@router.post(
    "/tune/rewrite-prompt",
    response_model=RewritePromptResponse,
    summary="Rewrite a system prompt to apply a tune cycle's changes",
)
async def assist_rewrite_prompt(req: RewritePromptRequest) -> RewritePromptResponse:
    prompt = rewrite_system_prompt(
        current_prompt=req.current_prompt,
        failure_type=req.failure_type,
        changes=req.changes,
        outcome_notes=req.outcome_notes,
    )
    text = await _ollama_generate(prompt, temperature=0.5)
    # Strip any accidental markdown fences the model may have added.
    return RewritePromptResponse(system_prompt=strip_code_fences(text))


# ---------------------------------------------------------------------------
# POST /ai/assist/tune/apply-plan
# ---------------------------------------------------------------------------

# Which ADLC phase each change_type is written to when a tune cycle is applied.
_CHANGE_PHASE = {
    "system_prompt": "build",
    "temperature": "build",
    "model": "build",
    "tools": "build",
    "definition": "define",
}


@router.post(
    "/tune/apply-plan",
    response_model=ApplyPlanResponse,
    summary="Build a multi-phase apply plan (Build config + Define) from a tune cycle",
)
async def assist_apply_plan(req: ApplyPlanRequest) -> ApplyPlanResponse:
    changes = [c for c in req.changes if isinstance(c, dict)]

    def of_type(t: str) -> list[dict]:
        return [c for c in changes if c.get("change_type") == t]

    # --- Build: rewrite the system prompt (plain-text call — robust on a small
    # context model; all changes are passed so behavioral fixes land in the prompt).
    rewrite_prompt = rewrite_system_prompt(
        current_prompt=req.current_prompt,
        failure_type=req.failure_type,
        changes=changes,
        outcome_notes=req.outcome_notes,
    )
    new_prompt = await _ollama_generate(rewrite_prompt, temperature=0.5)
    build = ApplyBuildPlan(system_prompt=strip_code_fences(new_prompt))

    # --- Build: temperature is extracted deterministically from the change text.
    for c in of_type("temperature"):
        temp = extract_temperature(c.get("description", ""))
        if temp is not None:
            build.temperature = temp
            break
    # model_name / tools are left for the user to confirm in the modal (pre-filled
    # with the current build values), since a small model rarely names them reliably.

    # --- Define: only if the cycle carries definition-level changes.
    definition: ApplyDefinitionPlan | None = None
    def_changes = of_type("definition")
    if def_changes:
        source = req.current_definition or req.agent_definition
        dplan_prompt = tune_definition_plan(source, def_changes, req.outcome_notes)
        raw = await _ollama_generate(dplan_prompt, temperature=0.4)
        parsed, had_error = extract_json(raw)
        if had_error or not isinstance(parsed, dict):
            salvaged = salvage_objects(
                raw,
                keys=("goals", "unsafe_zones", "constraints", "success_metrics", "intended_behaviors"),
            )
            parsed = salvaged[0] if salvaged else {}
        if isinstance(parsed, dict) and parsed:
            definition = ApplyDefinitionPlan(
                goals=parsed["goals"] if isinstance(parsed.get("goals"), str) else None,
                unsafe_zones=parsed["unsafe_zones"] if isinstance(parsed.get("unsafe_zones"), str) else None,
                constraints=parsed["constraints"] if isinstance(parsed.get("constraints"), list) else None,
                success_metrics=parsed["success_metrics"] if isinstance(parsed.get("success_metrics"), list) else None,
                intended_behaviors=parsed["intended_behaviors"] if isinstance(parsed.get("intended_behaviors"), list) else None,
            )

    # --- Targets: a human-readable map of every change to the phase it touches.
    targets = [
        ApplyTargetNote(
            phase=_CHANGE_PHASE.get(c.get("change_type", ""), "build"),
            change_type=c.get("change_type", "general"),
            description=c.get("description", ""),
        )
        for c in changes
    ]

    return ApplyPlanResponse(build=build, definition=definition, targets=targets)
