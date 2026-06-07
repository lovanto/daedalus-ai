from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import (
    ClassifyFailureRequest,
    ClassifyFailureResponse,
    EvalCaseSuggestion,
    RunEvalCaseRequest,
    RunEvalCaseResponse,
    SuggestEvalCasesRequest,
    SuggestEvalCasesResponse,
)
from ollama.client import OllamaClient, OllamaError, OllamaTimeoutError
from ollama.prompts import classify_failure, suggest_eval_cases
from utils import extract_json

router = APIRouter(prefix="/eval", tags=["eval"])


async def _generate(prompt: str, temperature: float = 0.5) -> str:
    client = OllamaClient()
    try:
        return await client.generate(prompt, temperature=temperature)
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /ai/eval/suggest-cases
# ---------------------------------------------------------------------------

@router.post(
    "/suggest-cases",
    response_model=SuggestEvalCasesResponse,
    summary="AI-suggested eval test cases based on agent definition",
)
async def suggest_cases(req: SuggestEvalCasesRequest) -> SuggestEvalCasesResponse:
    prompt = suggest_eval_cases(req.agent_definition.model_dump(), count=req.count)
    raw = await _generate(prompt, temperature=0.8)

    parsed, had_error = extract_json(raw)

    if had_error or not isinstance(parsed, list):
        # Try unwrapping {"cases": [...]} if model wrapped the array
        if isinstance(parsed, dict):
            inner = parsed.get("cases") or parsed.get("test_cases") or parsed.get("items")
            if isinstance(inner, list):
                parsed = inner
                had_error = False

    if had_error or not isinstance(parsed, list):
        return SuggestEvalCasesResponse(cases=[], parse_error=True)  # type: ignore[call-arg]

    cases = []
    for item in parsed:
        if isinstance(item, dict):
            cases.append(
                EvalCaseSuggestion(
                    input=item.get("input", ""),
                    expected_behavior=item.get("expected_behavior", ""),
                    category=item.get("category", "core"),
                )
            )

    return SuggestEvalCasesResponse(cases=cases)


# ---------------------------------------------------------------------------
# POST /ai/eval/classify-failure
# ---------------------------------------------------------------------------

@router.post(
    "/classify-failure",
    response_model=ClassifyFailureResponse,
    summary="Classify an eval failure as behavioral, structural, or scope",
)
async def classify_failure_endpoint(req: ClassifyFailureRequest) -> ClassifyFailureResponse:
    prompt = classify_failure(
        failure_description=req.failure_description,
        agent_definition=req.agent_definition.model_dump(),
    )
    raw = await _generate(prompt, temperature=0.3)

    parsed, had_error = extract_json(raw)

    if had_error or not isinstance(parsed, dict):
        return ClassifyFailureResponse(
            failure_type="behavioral",
            reasoning=raw,
            recommended_action="Review the raw AI response above",
        )

    return ClassifyFailureResponse(
        failure_type=parsed.get("failure_type", "behavioral"),
        reasoning=parsed.get("reasoning", ""),
        recommended_action=parsed.get("recommended_action", ""),
    )


# ---------------------------------------------------------------------------
# POST /ai/eval/run-case
# ---------------------------------------------------------------------------

@router.post(
    "/run-case",
    response_model=RunEvalCaseResponse,
    summary="Run a single eval test case against Ollama and score it",
)
async def run_eval_case(req: RunEvalCaseRequest) -> RunEvalCaseResponse:
    client = OllamaClient()

    # Step 1 — run the agent with the test input
    test_input = req.test_case.get("input", "")
    expected = req.test_case.get("expected_behavior", "")

    try:
        actual_output = await client.generate(
            prompt=test_input,
            system=req.system_prompt or None,
            model=req.model or None,
            temperature=0.7,
            use_cache=False,  # always re-run — reflect the current (e.g. post-tune) prompt
        )
    except OllamaTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Step 2 — evaluate: does the actual output satisfy the expected behavior?
    eval_prompt = f"""You are an AI evaluator. Decide whether the agent's response satisfies the expected behavior.

EXPECTED BEHAVIOR: {expected}

AGENT RESPONSE:
{actual_output}

Be strict: the response must actually satisfy the expected behavior, not just be related to it.

Respond ONLY with JSON:
{{
  "passed": true or false,
  "reasoning": "one or two sentences explaining the verdict"
}}"""

    try:
        eval_raw = await client.generate(eval_prompt, temperature=0.1, use_cache=False)
    except OllamaError:
        # Evaluation call failed — return the output without a verdict
        return RunEvalCaseResponse(
            passed=False,
            actual_output=actual_output,
            reasoning="Evaluation step failed — inspect the actual output manually.",
        )

    parsed, had_error = extract_json(eval_raw)

    if had_error or not isinstance(parsed, dict):
        return RunEvalCaseResponse(
            passed=False,
            actual_output=actual_output,
            reasoning=eval_raw,
        )

    return RunEvalCaseResponse(
        passed=bool(parsed.get("passed", False)),
        actual_output=actual_output,
        reasoning=parsed.get("reasoning", ""),
    )
