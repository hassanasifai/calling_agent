"""Generic, profile-driven call agent (task logic lives in agent profiles)."""

from __future__ import annotations

from pathlib import Path

from nrsp_agent.agent.engine import CallEngine
from nrsp_agent.agent.knowledge import KnowledgeBase
from nrsp_agent.agent.profile import load_profile


def build_engine(profile_dir: Path, call_data: dict[str, str]) -> CallEngine:
    """One call's engine from a profile directory and that call's data."""
    profile, documents = load_profile(profile_dir)
    return CallEngine(profile, KnowledgeBase(documents), dict(call_data))
