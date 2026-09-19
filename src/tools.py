from __future__ import annotations

import difflib
import os
import sys

from langchain_core.tools import tool

# saurav bhai parent directory ko sys.path me daal rhe hai taki mock_data
# alag alag entry point se bhi import ho jaye.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mock_data import mock_database

_CONTENT: dict = mock_database.get("content", {})

# saurav bhai index me sirf wahi topics dikhne chahiye jo sach me fetch ho sake.
# mock_data me `computer_vision` topics_index me pada hai par uska content hai hi
# nahi, to jo agent sahi ladder follow krega wo usi ko pick krega, "not found"
# milega, wapas index pe jayega, phir wahi dikhega aur loop lag jayega. isliye
# dono ka intersection hi serve kr rhe hai.
_SERVED_TOPICS: list[str] = [t for t in mock_database.get("topics_index", []) if t in _CONTENT]

_UNBACKED_TOPICS: list[str] = [
    t for t in mock_database.get("topics_index", []) if t not in _CONTENT
]


def _normalize(name: str) -> str:
    """Fold the cosmetic differences an LLM routinely introduces."""
    return (name or "").strip().lower().replace(" ", "_").replace("-", "_")


def _resolve(raw: str, candidates, what: str, scope: str = "") -> tuple[str | None, str | None]:
    """Map a possibly-hallucinated name onto a real key.

    Returns ``(canonical, None)`` or ``(None, error_text)``. The error always
    inlines the valid options so the model can correct itself on the next turn
    instead of guessing again and burning another loop iteration.
    """
    options = list(candidates)
    where = f" under '{scope}'" if scope else ""
    valid = f" Valid {what}s{where}: {', '.join(options) if options else '(none)'}."

    if not raw or not str(raw).strip():
        return None, f"Error: '{what}_name' is required but was empty.{valid}"

    if raw in options:
        return raw, None

    norm = _normalize(raw)
    for key in options:
        if _normalize(key) == norm:
            return key, None

    close = difflib.get_close_matches(norm, [_normalize(k) for k in options], n=1, cutoff=0.6)
    hint = ""
    if close:
        match = next(k for k in options if _normalize(k) == close[0])
        hint = f" Did you mean '{match}'?"
    return None, f"Error: {what.capitalize()} '{raw}' does not exist{where}.{hint}{valid}"


def resolve_topic(raw: str) -> tuple[str | None, str | None]:
    """Canonicalise a topic name against the served index."""
    return _resolve(raw, _SERVED_TOPICS, "topic")


def resolve_section(topic: str, raw: str) -> tuple[str | None, str | None]:
    """Canonicalise a section name within an already-resolved topic."""
    sections = _CONTENT.get(topic, {}).get("sections", {})
    return _resolve(raw, sections.keys(), "section", scope=topic)


@tool
def get_available_topics() -> str:
    """
    Use this tool FIRST when you need to answer a question but don't know what information is available.
    It returns a list of all high-level topic categories in the documentation.
    Cost: Very low context (Ideal for Layer 1 discovery).
    """
    return f"Available Topics Index: {', '.join(_SERVED_TOPICS)}"


@tool
def get_topic_summary(topic_name: str) -> str:
    """
    Use this tool to get a high-level summary of a specific topic.
    It also returns a list of detailed 'sections' available under this topic.
    Call this after you identify a relevant topic from the index.

    Args:
        topic_name: The exact name of the topic (e.g., 'deep_learning')
    """
    topic, err = resolve_topic(topic_name)
    if err:
        return err

    topic_data = _CONTENT[topic]
    summary = topic_data.get("summary", "No summary available.")
    available_sections = list(topic_data.get("sections", {}).keys())

    return (
        f"Topic: {topic}\n"
        f"Summary: {summary}\n"
        f"Available Detailed Sections to drill into: {', '.join(available_sections)}"
    )


@tool
def get_section_details(topic_name: str, section_name: str) -> str:
    """
    Use this tool to load the deep, technical details of a specific section.
    Only call this when the user needs specific instructions, code, or deep details
    that were not covered in the summary.

    Args:
        topic_name: The name of the parent topic.
        section_name: The specific section to load.
    """
    topic, err = resolve_topic(topic_name)
    if err:
        return err

    section, err = resolve_section(topic, section_name)
    if err:
        return err

    body = _CONTENT[topic]["sections"][section]
    return f"Detailed Content for {topic} -> {section}:\n{body}"


# saurav bhai in sabko ek list me rakh rhe hai taki graph.py me LLM se aasani se bind ho jaye
agent_tools = [get_available_topics, get_topic_summary, get_section_details]
