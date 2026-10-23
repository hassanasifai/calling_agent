"""Fake test call data for offline runs (never a real person)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NRSP_PROFILE = ROOT / "config" / "profiles" / "nrsp_verification"
TEST_CALL_DATA = {
    "customer_name": "احمد علی",
    "cnic_last4": "4567",
    "amount_spoken": "پانچ ہزار",
    "due_date_spoken": "پندرہ نومبر",
}
