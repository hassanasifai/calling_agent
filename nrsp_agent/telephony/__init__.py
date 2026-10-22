"""Stable telephony contracts and transport adapters."""

from nrsp_agent.telephony.contracts import (
    AudioFrame,
    CallEvent,
    CallEventType,
    CallHandle,
    TelephonyPort,
)
from nrsp_agent.telephony.diago import DiagoTelephonyPort

__all__ = [
    "AudioFrame",
    "CallEvent",
    "CallEventType",
    "CallHandle",
    "DiagoTelephonyPort",
    "TelephonyPort",
]
