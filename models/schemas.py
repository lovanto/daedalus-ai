from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class AIResult(BaseModel):
    result: str
    parse_error: bool = False


class AgentDefinitionPayload(BaseModel):
    goals: str = ""
    intended_behaviors: Any = None
    constraints: Any = None
    success_metrics: Any = None
    unsafe_zones: str = ""
    confidence_threshold: float = 75.0
    sops: str = ""


# ---------------------------------------------------------------------------
# /ai/assist/define
# ---------------------------------------------------------------------------

class AssistDefineRequest(BaseModel):
    section: str = Field(..., description="mission|behaviors|constraints|metrics|sops")
    agent_name: str
    agent_description: str = ""
    existing_content: str = ""
    think: bool = Field(
        default=False,
        description="Return the model's reasoning trace. Slower — use for interactive single-section assist, not batch drafts.",
    )


class DefineSuggestion(BaseModel):
    title: str
    content: str
    reasoning: str


class AssistDefineResponse(BaseModel):
    suggestions: list[DefineSuggestion]
    thinking: str = ""


# ---------------------------------------------------------------------------
# /ai/assist/build/system-prompt
# ---------------------------------------------------------------------------

class BuildSystemPromptRequest(BaseModel):
    agent_definition: AgentDefinitionPayload


class BuildSystemPromptResponse(BaseModel):
    system_prompt: str


# ---------------------------------------------------------------------------
# /ai/assist/eval/suggest-cases
# ---------------------------------------------------------------------------

class SuggestEvalCasesRequest(BaseModel):
    agent_definition: AgentDefinitionPayload
    count: int = Field(default=5, ge=1, le=15)


class EvalCaseSuggestion(BaseModel):
    input: str
    expected_behavior: str
    category: str = Field(..., description="core|edge_case|adversarial|regression")


class SuggestEvalCasesResponse(BaseModel):
    cases: list[EvalCaseSuggestion]
    parse_error: bool = False


# ---------------------------------------------------------------------------
# /ai/eval/classify-failure
# ---------------------------------------------------------------------------

class ClassifyFailureRequest(BaseModel):
    failure_description: str
    agent_definition: AgentDefinitionPayload


class ClassifyFailureResponse(BaseModel):
    failure_type: str = Field(..., description="behavioral|structural|scope")
    reasoning: str
    recommended_action: str


# ---------------------------------------------------------------------------
# /ai/eval/run-case
# ---------------------------------------------------------------------------

class RunEvalCaseRequest(BaseModel):
    test_case: dict
    system_prompt: str
    model: str = ""


class RunEvalCaseResponse(BaseModel):
    passed: bool
    actual_output: str
    reasoning: str


# ---------------------------------------------------------------------------
# /ai/chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="user|assistant")
    content: str


class ChatRequest(BaseModel):
    # Full conversation so far (oldest first), including the latest user turn.
    messages: list[ChatMessage] = Field(default_factory=list)
    # The agent's persona — injected by the Go proxy from the latest build.
    system_prompt: str = ""
    model: str = ""


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# /ai/analyze/patterns
# ---------------------------------------------------------------------------

class AnalyzePatternsRequest(BaseModel):
    observations: list[str]
    agent_name: str = ""


class AnalyzePatternsResponse(BaseModel):
    key_patterns: list[str]
    severity_summary: str
    recommended_route: str
    recommended_route_reasoning: str = ""


# ---------------------------------------------------------------------------
# /ai/analyze/scope-drift
# ---------------------------------------------------------------------------

class ScopeDriftRequest(BaseModel):
    observations: list[str]
    agent_definition: AgentDefinitionPayload


class ScopeDriftResponse(BaseModel):
    drift_detected: bool
    drift_areas: list[str]
    drift_severity: str = ""
    recommendation: str


# ---------------------------------------------------------------------------
# /ai/assist/tune/suggest-fix
# ---------------------------------------------------------------------------

class TuneFixFailingCase(BaseModel):
    input: str = ""
    expected_behavior: str = ""
    actual_output: str = ""
    reasoning: str = ""
    category: str = ""


class TuneFixEvalRecord(BaseModel):
    score: float = 0.0
    failure_type: str = ""
    test_cases_passed: int = 0
    test_cases_failed: int = 0
    notes: str = ""
    source: str = ""
    created_at: str = ""


class TuneFixRequest(BaseModel):
    failure_type: str = Field(..., description="behavioral|structural|scope")
    failure_description: str
    current_build: dict = {}
    # Recent eval history (newest first) — lets the model read the score trend
    # and whether the same failure type keeps recurring.
    eval_history: list[TuneFixEvalRecord] = Field(default_factory=list)
    # The concrete cases that failed in the latest eval, with the agent's actual
    # output and the judge's verdict — the strongest signal for a targeted fix.
    failing_cases: list[TuneFixFailingCase] = Field(default_factory=list)


class TuneFixSuggestion(BaseModel):
    change_type: str
    description: str
    expected_impact: str


class TuneFixResponse(BaseModel):
    suggestions: list[TuneFixSuggestion]


# ---------------------------------------------------------------------------
# /ai/assist/tune/rewrite-prompt
# ---------------------------------------------------------------------------

class RewritePromptRequest(BaseModel):
    current_prompt: str = ""
    failure_type: str = Field(default="", description="behavioral|structural|scope|none")
    changes: list[dict] = Field(
        default_factory=list,
        description="The tune cycle's changes: [{change_type, description, expected_impact}]",
    )
    outcome_notes: str = ""


class RewritePromptResponse(BaseModel):
    system_prompt: str


# ---------------------------------------------------------------------------
# /ai/assist/tune/apply-plan
# ---------------------------------------------------------------------------

class ApplyPlanRequest(BaseModel):
    current_prompt: str = ""
    failure_type: str = Field(default="", description="behavioral|structural|scope|none")
    changes: list[dict] = Field(
        default_factory=list,
        description="The tune cycle's changes: [{change_type, description, expected_impact}]",
    )
    outcome_notes: str = ""
    # Current build config the plan should start from (FE supplies this).
    current_build: dict = Field(default_factory=dict)
    # Current definition — supplied by FE as current_definition, or auto-injected
    # by the Go proxy as agent_definition. Either is accepted.
    current_definition: dict = Field(default_factory=dict)
    agent_definition: dict = Field(default_factory=dict)


class ApplyBuildPlan(BaseModel):
    system_prompt: str = ""
    model_provider: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    tools: list | None = None


class ApplyDefinitionPlan(BaseModel):
    goals: str | None = None
    constraints: list | None = None
    unsafe_zones: str | None = None
    success_metrics: list | None = None
    intended_behaviors: list | None = None


class ApplyTargetNote(BaseModel):
    phase: str  # "build" | "define"
    change_type: str
    description: str


class ApplyPlanResponse(BaseModel):
    build: ApplyBuildPlan
    definition: ApplyDefinitionPlan | None = None
    targets: list[ApplyTargetNote] = Field(default_factory=list)
