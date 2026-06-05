"""Shared helpers used across routers."""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> tuple[dict | list | None, bool]:
    """
    Try to extract parseable JSON from LLM output.
    Returns (parsed_value, had_error).
    Handles markdown code fences and leading/trailing prose.
    """
    # Direct parse
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1)), False
        except json.JSONDecodeError:
            pass

    # Find outermost JSON object or array by scanning for braces/brackets
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1]), False
            except json.JSONDecodeError:
                pass

    return None, True
