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


def strip_code_fences(text: str) -> str:
    """Drop a leading ```lang fence and trailing ``` fence, if present."""
    text = text.strip()
    text = re.sub(r"^```[^\n]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_temperature(text: str) -> float | None:
    """
    Pull a plausible temperature value (a decimal in [0, 2]) out of a freeform
    change description such as "Decrease temperature to 0.5". Only decimals are
    accepted so we don't misread stray integers like "add 2 tools".
    """
    for m in re.finditer(r"\b([0-2]\.\d+)\b", text):
        val = float(m.group(1))
        if 0.0 <= val <= 2.0:
            return val
    return None


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


def coerce_suggestions(
    parsed,
    raw: str,
    keys: tuple[str, ...] = (),
    text_key: str | None = None,
) -> list[dict]:
    """
    Normalize whatever the model returned into a list of suggestion dicts.

    Models are supposed to return ``{"suggestions": [ {...}, ... ]}`` but in
    practice also emit a few malformed shapes that ``extract_json`` parses into
    the wrong container:

    * a bare list of suggestion dicts;
    * a JSON array whose elements are the *lines* of the intended JSON object
      (``["{", "\\"suggestions\\": [", "{", ...]``) — rejoining the lines and
      re-parsing recovers the real object;
    * a JSON array of plain strings, where each string is itself a suggestion.

    Returns a list of dicts (possibly empty). ``text_key`` controls how a bare
    string is wrapped into a dict when no structured object is available.
    """

    def container_items(val):
        if isinstance(val, dict):
            inner = val.get("suggestions")
            if isinstance(inner, list):
                return inner
            # A single suggestion object returned without the wrapper.
            if keys and any(k in val for k in keys):
                return [val]
            return []
        if isinstance(val, list):
            return val
        return []

    items = container_items(parsed)

    dicts = [s for s in items if isinstance(s, dict)]
    if dicts:
        return dicts

    # Items are strings: either the JSON object split across array elements, or
    # genuine one-line suggestions. Try rejoining + re-parsing first.
    strings = [s for s in items if isinstance(s, str) and s.strip()]
    if strings:
        rejoined, err = extract_json("\n".join(strings))
        if not err and rejoined is not None:
            inner_dicts = [s for s in container_items(rejoined) if isinstance(s, dict)]
            if inner_dicts:
                return inner_dicts
        if text_key:
            return [{text_key: s.strip()} for s in strings]

    # Last resort: scrape balanced objects out of the raw text.
    return salvage_objects(raw, keys=keys)


def salvage_objects(text: str, keys: tuple[str, ...] = ()) -> list[dict]:
    """
    Best-effort recovery of complete JSON objects from malformed or truncated
    LLM output.

    Scans for balanced ``{...}`` spans (string- and escape-aware), parses each
    leniently, and keeps the dicts that contain at least one of ``keys`` (if
    given). Incomplete objects — e.g. the final element when the model is cut
    off mid-stream — are simply skipped instead of poisoning the whole parse.
    """
    text = strip_think(text)
    objects: list[dict] = []
    stack: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                val = _loads_lenient(text[start : i + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(val, dict) and (not keys or any(k in val for k in keys)):
                objects.append(val)
    return objects
