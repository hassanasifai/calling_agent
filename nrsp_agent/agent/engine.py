"""Generic call engine: executes a profile's actions safely.

The LLM chooses one action per caller turn (or ``reply``) and writes the
natural reply; this engine validates the choice against the profile
(required/blocking flags), runs consequential checks in code, and decides what
is spoken: an approved profile text, or the model's guarded reply followed by
the pending approved question so the call returns to its objective.

Sensitive values are compared only with digits parsed by code from the
caller's own words, read back for confirmation; the model never sees private
call data and its reconstruction of numbers is never trusted.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass, field

from nrsp_agent.agent.knowledge import KnowledgeBase
from nrsp_agent.agent.profile import Action, ActionKind, AgentProfile
from nrsp_agent.agent.textnorm import extract_digits, spoken_digits

REPLY = "reply"
AFFIRM = "affirm"
REJECT = "reject"
IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class Decision:
    """What to say for one turn.

    Exactly one of ``speak`` (approved text) or ``use_model_reply`` is set;
    ``append`` (approved) follows a model reply.
    """

    speak: str = ""
    use_model_reply: bool = False
    append: str = ""
    end_call: bool = False
    asr_prompt: str | None = None


class _Formatter(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"approved text references unknown field {{{key}}}")


class _Placeholder(dict):
    """Renders ``{field}`` as ``[field]`` so the prompt is call-independent."""

    def __missing__(self, key: str) -> str:
        return f"[{key}]"


@dataclass
class CallEngine:
    profile: AgentProfile
    knowledge: KnowledgeBase
    call_data: dict[str, str]
    flags: set[str] = field(default_factory=set)
    fields: dict[str, str] = field(default_factory=dict)
    outcome: str = IN_PROGRESS
    pending_ask: str = ""

    def __post_init__(self) -> None:
        missing = [f for f in self.profile.public_fields if f not in self.call_data]
        if missing:
            raise ValueError(f"call data lacks public fields {missing}")
        self._actions = {a.name: a for a in self.profile.actions}
        texts = {
            k: v.format_map(_Formatter(self.call_data))
            for k, v in self.profile.texts.items()
        }
        self._values = _Formatter({**self.call_data, **texts})
        self._turns = self._silences = self._failed = 0
        self._attempts: dict[str, int] = {}
        # A pending confirmation: (verify action, digits read back) or
        # (end action, "") awaiting the caller's affirm/reject.
        self._candidate: tuple[Action, str] | None = None
        self._return_question = ""
        self._verify_question = ""
        # The greeting's question is pending from the start of the call.
        self.pending_ask = self.render(self.profile.greeting_ask)

    # ----- approved texts -------------------------------------------------
    def render(self, template: str) -> str:
        return template.format_map(self._values).strip()

    def secrets(self) -> list[str]:
        """Private call-data values that must never reach the model."""
        public = set(self.profile.public_fields)
        return [v for k, v in self.call_data.items() if k not in public and v]

    @property
    def closed(self) -> bool:
        return self.outcome != IN_PROGRESS

    def greeting(self) -> Decision:
        return self._approved(self.profile.greeting, self.profile.greeting_ask)

    # ----- model I/O ------------------------------------------------------
    def action_names(self) -> list[str]:
        names = [REPLY, *self._actions]
        if any(
            (a.kind == ActionKind.VERIFY and a.readback) or a.confirm
            for a in self.profile.actions
        ):
            names += [AFFIRM, REJECT]
        return names

    def allowed_actions(self) -> list[str]:
        if self._candidate:
            # A yes/no confirmation is pending: accept the answer, a new
            # attempt at the same check, or a profile safety exit only.
            pending, _ = self._candidate
            keep = [
                a.name
                for a in self._available()
                if (a.name in self.profile.safety_actions and a is not pending)
                or (pending.kind == ActionKind.VERIFY and a is pending)
            ]
            return [AFFIRM, REJECT, *keep]
        return [REPLY] + [a.name for a in self._available()]

    def response_schema(self) -> dict:
        """JSON schema for constrained decoding of this turn.

        Only actions allowed now can be produced, and ``action`` comes first
        so the decision is known after a few tokens. Actions answered by an
        approved text must have an empty ``say``, so generation stops at once
        instead of spending CPU on words that would be discarded.
        """
        allowed = self.allowed_actions()
        talking = [n for n in allowed if self._model_speaks(n)]
        silent = [n for n in allowed if not self._model_speaks(n)]

        def branch(names: list[str], say: dict) -> dict:
            return {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": names},
                    "value": {"type": "string", "maxLength": 80},
                    "say": say,
                },
                "required": ["action", "value", "say"],
                "additionalProperties": False,
            }

        branches = []
        if talking:
            branches.append(branch(talking, {"type": "string", "maxLength": 300}))
        if silent:
            branches.append(branch(silent, {"type": "string", "maxLength": 0}))
        return branches[0] if len(branches) == 1 else {"anyOf": branches}

    def needs_value(self, name: str) -> bool:
        """Whether the action uses the model's ``value`` (record actions)."""
        action = self._actions.get(name)
        return action is not None and action.kind == ActionKind.RECORD

    def _model_speaks(self, name: str) -> bool:
        action = self._actions.get(name)
        return name == REPLY or (
            action is not None and action.kind == ActionKind.RECORD
        )

    def system_prompt(self) -> str:
        """Identical for every call of a profile except the final call-details
        lines, so the server's prefix cache is reused across calls (measured:
        a cold ~1,240-token prefix costs ~35 s on 4-5 CPU threads)."""
        p = self.profile

        def static(template: str) -> str:
            return template.format_map(_Placeholder()).strip()

        facts = "\n".join(f"- {k}: {self.call_data[k]}" for k in p.public_fields)
        rules = "\n".join(f"- {r}" for r in p.rules)
        if not p.reply_allow_digits:
            rules += (
                '\n- In "say" never state numbers, amounts or dates; when such a'
                " fact is needed choose the action that provides it."
            )
        actions = "\n".join(
            f"- {a.name}: {static(a.when)}"
            + (
                ' [system speaks approved text; say ""]'
                if a.kind != ActionKind.RECORD
                and (a.say or a.kind == ActionKind.VERIFY)
                else ""
            )
            for a in p.actions
        )
        builtins = (
            f"- {AFFIRM}: the caller says yes to your confirmation question"
            " (digits you read back, or confirming their request)\n"
            f"- {REJECT}: the caller says no to your confirmation question\n"
            if AFFIRM in self.action_names()
            else ""
        )
        examples = "\n".join(
            (f"({static(e.context)}) " if e.context else "")
            + f"Caller: {static(e.caller)} => "
            + json.dumps(
                {"action": e.action, "value": e.value, "say": static(e.say)},
                ensure_ascii=False,
            )
            for e in p.examples
        )
        if examples:
            examples = f"\n\nExamples:\n{examples}"
        return (
            f"{static(p.persona)}\n\nTask: {static(p.objective)}\n\n"
            f"Rules:\n{rules}\n\n"
            "Callers may speak Urdu, Roman Urdu, English or a mix; their words come"
            " from speech recognition and may be misspelled, so infer the meaning."
            "\n\nFor each caller turn choose exactly one action:\n"
            f"- {REPLY}: answer, clarify or respond naturally when no other action"
            f" fits\n{actions}\n{builtins}\n"
            'Answer only with JSON {"action": ..., "value": ..., "say": ...}.'
            ' "value": the detail to record for record actions, else "".'
            ' "say": your words to the caller in natural, polite spoken Urdu script,'
            " at most two short sentences, without asking the next question"
            " (the system asks it)."
            f"{examples}"
            f"\n\nThis call's details (the [names] above refer to these):\n{facts}"
        )

    def turn_message(self, caller_text: str, redact) -> str:
        """Per-turn dynamic context, kept short: every token here is processed
        on each turn (measured ~43 prompt tokens/s for a 2B model on 4 CPU
        threads). Allowed actions are enforced by the schema, not listed."""
        done = ", ".join(sorted(self.flags)) or "nothing yet"
        parts = [f"[done: {done}]"]
        if self.pending_ask:
            parts[0] = f"[done: {done} | you asked: {redact(self.pending_ask)}]"
        if self._candidate:
            parts.append(
                "[yes/no question pending: affirm if the caller means yes,"
                " reject if no]"
            )
        for snippet in self.knowledge.search(caller_text):
            parts.append(f"Knowledge: {snippet[:400]}")
        parts.append(f"Caller: {caller_text}")
        return "\n".join(parts)

    # ----- decisions ------------------------------------------------------
    def on_model(self, action: str, value: str, caller_text: str) -> Decision | None:
        if self.closed:
            return None
        self._silences = 0
        self._failed = 0
        self._turns += 1
        if self._turns > self.profile.max_turns:
            return self._end(
                "max_turns", self.profile.max_turns_close or self.profile.unclear_close
            )
        if self._candidate and action in (AFFIRM, REJECT):
            return self._resolve_candidate(action == AFFIRM)
        if self._candidate and action not in self.allowed_actions():
            # Not an answer to the confirmation: ask it again.
            return self._approved(self.profile.repeat_request, self.pending_ask)
        chosen = self._actions.get(action)
        if chosen is None or chosen not in self._available():
            return self._model_reply()
        if chosen.kind == ActionKind.VERIFY:
            return self._verify(chosen, caller_text)
        self._candidate = None
        if chosen.kind == ActionKind.END:
            if chosen.confirm:
                self._candidate = (chosen, "")
                self._return_question = self.pending_ask
                self.pending_ask = self.render(chosen.confirm)
                return Decision(speak=self.pending_ask)
            return self._finish(chosen)
        self.flags.update(chosen.sets)
        if chosen.kind == ActionKind.RECORD:
            self.fields[chosen.field] = (value or caller_text).strip()[:200]
            return self._model_reply(chosen.ask, chosen.asr_prompt)
        return self._approved(chosen.say, chosen.ask, chosen.asr_prompt)

    def on_keypad(self, digits: str) -> Decision | None:
        """Keypad (DTMF) entry for the verification that is currently open.

        Keypad digits are exact, so they are compared without read-back.
        Returns ``None`` when no verification is waiting for digits.
        """
        if self.closed:
            return None
        action = next(
            (a for a in self._available() if a.kind == ActionKind.VERIFY), None
        )
        if action is None:
            return None
        self._silences = self._failed = 0
        self._candidate = None
        if not self._verify_question:
            self._verify_question = self.pending_ask
        if len(digits) != action.digits:
            return self._approved(action.say_unclear, self._verify_ask(action))
        return self._check(action, digits)

    def keypad_length(self) -> int | None:
        """Digits expected now from the keypad, if a verification is open."""
        for a in self._available():
            if a.kind == ActionKind.VERIFY:
                return a.digits
        return None

    def on_failure(self) -> Decision | None:
        """The model gave no usable answer (timeout, invalid output, ASR loop)."""
        if self.closed:
            return None
        self._failed += 1
        if self._failed > self.profile.max_failed_turns:
            return self._end("unclear", self.profile.unclear_close)
        return self._approved(self.profile.repeat_request, self.pending_ask)

    def on_silence(self, finish_action: str = "") -> Decision | None:
        if self.closed:
            return None
        finish = self._actions.get(finish_action)
        if finish and set(self.profile.completion) <= self.flags:
            return self._end(finish.outcome, finish.say)
        self._silences += 1
        if self._silences > self.profile.max_silence_prompts:
            return self._end("no_response", self.profile.no_response)
        return self._approved(self.profile.silence_prompt, self.pending_ask)

    def result(self) -> dict:
        return {
            "profile": self.profile.name,
            "outcome": self.outcome,
            "flags": sorted(self.flags),
            "fields": dict(self.fields),
        }

    # ----- internals ------------------------------------------------------
    def _available(self) -> list[Action]:
        return [
            a
            for a in self.profile.actions
            if set(a.requires) <= self.flags and not set(a.blocked_by) & self.flags
        ]

    def _approved(self, say: str, ask: str = "", asr_prompt: str = "") -> Decision:
        if ask:
            self.pending_ask = self.render(ask)
        text = " ".join(x for x in (self.render(say), self.render(ask)) if x)
        return Decision(speak=text, asr_prompt=self._asr(asr_prompt))

    def _model_reply(self, ask: str = "", asr_prompt: str = "") -> Decision:
        if ask:
            self.pending_ask = self.render(ask)
        return Decision(
            use_model_reply=True,
            append=self.pending_ask,
            asr_prompt=self._asr(asr_prompt),
        )

    def _end(self, outcome: str, say: str) -> Decision:
        self.outcome = outcome
        self._candidate = None
        return Decision(speak=self.render(say), end_call=True)

    def _asr(self, name: str) -> str | None:
        return self.profile.asr_prompts.get(name) if name else None

    def _verify(self, action: Action, caller_text: str) -> Decision:
        # Digits come only from the caller's transcript, parsed by code.
        if not self._candidate:
            self._verify_question = self.pending_ask
        digits = extract_digits(caller_text)
        if action.digits < len(digits) <= action.digits + 2 and action.readback:
            # ASR often adds a count word ("آخری چار ہندسے"): read back the
            # trailing digits and let the caller confirm or correct.
            digits = digits[-action.digits :]
        if len(digits) != action.digits:
            self._candidate = None
            return self._approved(action.say_unclear, self._verify_ask(action))
        if action.readback:
            self._candidate = (action, digits)
            readback = self.profile.readback.replace(
                "{digits_spoken}", spoken_digits(digits)
            )
            self.pending_ask = self.render(readback)
            return Decision(speak=self.pending_ask)
        return self._check(action, digits)

    def _finish(self, action: Action) -> Decision:
        outcome = action.outcome
        if action.outcome_if_incomplete and not set(self.profile.completion) <= (
            self.flags
        ):
            outcome = action.outcome_if_incomplete
        return self._end(outcome, action.say)

    def _resolve_candidate(self, confirmed: bool) -> Decision:
        assert self._candidate is not None
        action, digits = self._candidate
        self._candidate = None
        if action.kind == ActionKind.END:
            if confirmed:
                return self._finish(action)
            question = self._return_question.replace("{", "{{").replace("}", "}}")
            return self._approved("", question)
        if not confirmed:
            return self._approved(action.say_unclear, self._verify_ask(action))
        return self._check(action, digits)

    def _check(self, action: Action, digits: str) -> Decision:
        expected = self.call_data.get(action.field, "")
        if expected and hmac.compare_digest(digits, expected):
            self.flags.update(action.sets)
            return self._approved(action.say, action.ask, action.asr_prompt)
        self._attempts[action.name] = self._attempts.get(action.name, 0) + 1
        if self._attempts[action.name] >= action.max_attempts:
            return self._end(
                action.outcome_failed or "verification_failed", action.say_failed
            )
        return self._approved(action.say_retry, self._verify_ask(action))

    def _verify_ask(self, action: Action) -> str:
        # Re-ask whatever question led to this verification.
        question = self._verify_question or self.pending_ask
        return question.replace("{", "{{").replace("}", "}}")
