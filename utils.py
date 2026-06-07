"""Shared helpers used across routers."""
from __future__ import annotations

import json
import re

# Matches a qwen3-style reasoning block that some models inline into the answer.
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
# Trailing comma before a closing brace/bracket — a very common LLM JSON slip.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_think(text: str) -> str:
    """Remove any inline <think>…</think> reasoning block from model output."""
    return _THINK_BLOCK.sub("", text).strip()


def _loads_lenient(snippet: str):
    """json.loads with a couple of forgiving repairs for common LLM mistakes."""
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass
    # Drop trailing commas, e.g. `[1, 2, ]` or `{"a": 1, }`
    repaired = _TRAILING_COMMA.sub(r"\1", snippet)
    return json.loads(repaired)  # may still raise — caller handles it


def extract_json(text: str) -> tuple[dict | list | None, bool]:
    """
    Try to extract parseable JSON from LLM output.
    Returns (parsed_value, had_error).
    Handles inline reasoning blocks, markdown code fences, leading/trailing
    prose, and common JSON slips like trailing commas.
    """
    text = strip_think(text)

    # Direct parse
    try:
        return _loads_lenient(text), False
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return _loads_lenient(m.group(1)), False
        except json.JSONDecodeError:
            pass

    # Find outermost JSON object or array by scanning for braces/brackets
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return _loads_lenient(text[start : end + 1]), False
            except json.JSONDecodeError:
                pass

    return None, True
