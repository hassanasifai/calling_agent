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
        self._candidate: tuple[Action, str] | None = None
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
            a.kind == ActionKind.VERIFY and a.readback for a in self.profile.actions
        ):
            names += [AFFIRM, REJECT]
        return names

    def allowed_actions(self) -> list[str]:
        names = [REPLY] + [a.name for a in self._available()]
        return names + [AFFIRM, REJECT] if self._candidate else names

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

        branches = [branch(talking, {"type": "string", "maxLength": 300})]
        if silent:
            branches.append(branch(silent, {"type": "string", "maxLength": 0}))
        return branches[0] if len(branches) == 1 else {"anyOf": branches}

    def _model_speaks(self, name: str) -> bool:
        action = self._actions.get(name)
        return name == REPLY or (
            action is not None and action.kind == ActionKind.RECORD
        )

    def system_prompt(self) -> str:
        """Static for the whole call, so the server's prefix cache covers it."""
        p = self.profile
        facts = "\n".join(f"- {k}: {self.call_data[k]}" for k in p.public_fields)
        rules = "\n".join(f"- {r}" for r in p.rules)
        if not p.reply_allow_digits:
            rules += (
                '\n- In "say" never state numbers, amounts or dates; when such a'
                " fact is needed choose the action that provides it."
            )
        actions = "\n".join(
            f"- {a.name}: {self.render(a.when)}"
            + (
                ' [system speaks approved text; say ""]'
                if a.kind != ActionKind.RECORD
                and (a.say or a.kind == ActionKind.VERIFY)
                else ""
            )
            for a in p.actions
        )
        builtins = (
            f"- {AFFIRM}: the caller confirms the digits you read back\n"
            f"- {REJECT}: the caller says the digits you read back are wrong\n"
            if AFFIRM in self.action_names()
            else ""
        )
        examples = "\n".join(
            (f"({self.render(e.context)}) " if e.context else "")
            + f"Caller: {self.render(e.caller)} => "
            + json.dumps(
                {"action": e.action, "value": e.value, "say": self.render(e.say)},
                ensure_ascii=False,
            )
            for e in p.examples
        )
        if examples:
            examples = f"\n\nExamples:\n{examples}"
        return (
            f"{p.persona}\n\nTask: {self.render(p.objective)}\n\n"
            f"Known call details:\n{facts}\n\nRules:\n{rules}\n\n"
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
        )

    def turn_message(self, caller_text: str, redact) -> str:
        """Per-turn dynamic context, kept short to limit prompt processing."""
        allowed = self.allowed_actions()
        parts = [
            f"Done: {', '.join(sorted(self.flags)) or 'nothing yet'}."
            f" Allowed actions now: {', '.join(allowed)}."
        ]
        if self.pending_ask:
            parts.append(f"You last asked: {redact(self.pending_ask)}")
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
        chosen = self._actions.get(action)
        if chosen is None or chosen not in self._available():
            return self._model_reply()
        if chosen.kind == ActionKind.VERIFY:
            return self._verify(chosen, caller_text)
        self._candidate = None
        if chosen.kind == ActionKind.END:
            outcome = chosen.outcome
            if chosen.outcome_if_incomplete and not set(self.profile.completion) <= (
                self.flags
            ):
                outcome = chosen.outcome_if_incomplete
            return self._end(outcome, chosen.say)
        self.flags.update(chosen.sets)
        if chosen.kind == ActionKind.RECORD:
            self.fields[chosen.field] = (value or caller_text).strip()[:200]
            return self._model_reply(chosen.ask, chosen.asr_prompt)
        return self._approved(chosen.say, chosen.ask, chosen.asr_prompt)

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

    def _resolve_candidate(self, confirmed: bool) -> Decision:
        assert self._candidate is not None
        action, digits = self._candidate
        self._candidate = None
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
