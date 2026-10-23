"""Agent profile: everything task-specific about a call, loaded from config.

A profile directory holds ``profile.toml`` and optional knowledge files. The
engine never contains task wording, fields or rules; switching task means
switching profile. Approved texts may reference call-data fields and other
texts as ``{name}``.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionKind(StrEnum):
    # Speak the action's approved text (optionally gated by flags).
    SAY = "say"
    # Store the model's extracted value in a result field; speak its reply.
    RECORD = "record"
    # Compare digits parsed (by code) from the caller's words to private data.
    VERIFY = "verify"
    # Speak the approved text and end the call with an outcome.
    END = "end"


class Action(BaseModel):
    """One thing the model may decide to do; code validates and executes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    kind: ActionKind
    when: str = Field(min_length=3, description="When to choose it (for the model)")
    requires: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    sets: tuple[str, ...] = ()
    say: str = ""
    # Approved question that the agent asks next and waits on.
    ask: str = ""
    outcome: str = ""
    # END only: approved question asked first; the call ends only if the
    # caller confirms (guards against misunderstood, irreversible endings).
    confirm: str = ""
    # END only: outcome when the profile's completion flags are not all set.
    outcome_if_incomplete: str = ""
    field: str = ""
    asr_prompt: str = ""
    # VERIFY only.
    digits: int = Field(default=4, ge=1, le=12)
    max_attempts: int = Field(default=2, ge=1, le=5)
    readback: bool = True
    say_retry: str = ""
    say_unclear: str = ""
    say_failed: str = ""
    outcome_failed: str = ""

    @model_validator(mode="after")
    def _check(self) -> Action:
        if self.kind == ActionKind.END and not self.outcome:
            raise ValueError(f"end action {self.name} needs an outcome")
        if self.kind in (ActionKind.VERIFY, ActionKind.RECORD) and not self.field:
            raise ValueError(f"{self.kind} action {self.name} needs a field")
        if self.kind == ActionKind.VERIFY and not (
            self.say_retry and self.say_unclear and self.say_failed
        ):
            raise ValueError(f"verify action {self.name} needs retry/unclear/failed")
        return self


class Example(BaseModel):
    """A worked caller turn shown to the model (in the cached system prompt)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    caller: str
    action: str
    value: str = ""
    say: str = ""
    # Flags assumed set when this example happens (context for the model).
    context: str = ""


class AgentProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    language: str = "ur"
    persona: str
    objective: str
    rules: tuple[str, ...] = ()
    # Call-data fields the model may see; all other fields stay private.
    public_fields: tuple[str, ...] = ()
    # Actions that stay available even while a yes/no confirmation is
    # pending (e.g. opt-out, human handoff); others wait for the answer.
    safety_actions: tuple[str, ...] = ()
    # Flags that make an END action count as a completed task.
    completion: tuple[str, ...] = ()
    greeting: str
    greeting_ask: str = ""
    texts: dict[str, str] = Field(default_factory=dict)
    actions: tuple[Action, ...]
    examples: tuple[Example, ...] = ()
    asr_prompts: dict[str, str] = Field(default_factory=dict)
    knowledge_files: tuple[str, ...] = ()
    reply_forbidden_terms: tuple[str, ...] = ()
    reply_allow_digits: bool = False
    max_turns: int = Field(default=16, ge=2, le=100)
    max_silence_prompts: int = Field(default=2, ge=0, le=5)
    max_failed_turns: int = Field(default=2, ge=1, le=5)
    # Approved texts the engine itself needs.
    silence_prompt: str
    no_response: str
    repeat_request: str
    unclear_close: str
    readback: str = ""
    max_turns_close: str = ""

    @model_validator(mode="after")
    def _unique_actions(self) -> AgentProfile:
        names = [a.name for a in self.actions]
        if len(names) != len(set(names)) or "reply" in names:
            raise ValueError("action names must be unique and not 'reply'")
        if any(a.kind == ActionKind.VERIFY and a.readback for a in self.actions) and (
            not self.readback
        ):
            raise ValueError("a verify action with readback needs the readback text")
        return self


def load_profile(directory: Path) -> tuple[AgentProfile, list[tuple[str, str]]]:
    """Load ``profile.toml`` and its knowledge files as (source, text) pairs."""
    profile = AgentProfile.model_validate(
        tomllib.loads((directory / "profile.toml").read_text(encoding="utf-8"))
    )
    knowledge = [
        (name, (directory / name).read_text(encoding="utf-8"))
        for name in profile.knowledge_files
    ]
    return profile, knowledge
