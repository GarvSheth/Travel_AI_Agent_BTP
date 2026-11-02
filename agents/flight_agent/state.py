"""LangGraph state for the flight recommendation workflow.

Everything stored here is plain data (str / int / float / list / dict) so the
checkpointer can persist it across clarification interrupts.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class FlightState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────
    origin: str                          # IATA code, e.g. "BOM"
    destination: str                     # IATA code, e.g. "LHR"
    departure_date: str                  # "YYYY-MM-DD"
    return_date: Optional[str]           # "YYYY-MM-DD" or None for one-way
    num_passengers: int
    budget_usd: Optional[float]
    currency: str                        # ISO 4217, e.g. "USD", "INR"
    preferences: str                     # free-form text from input pipeline

    # ── Clarification ────────────────────────────────────────────────────
    cabin_class: Optional[str]           # None = not asked yet; "" = no preference
    max_stops: Optional[int]             # None = not asked yet; 2 = any
    preferred_departure_period: Optional[str]   # "morning"|"afternoon"|"evening"|"any"
    clarification_question: Optional[str]
    clarification_required: bool
    clarification_attempts: int          # questions asked so far
    clarification_history: list[dict]    # [{question, answer}] shown to the LLM
    preference_answer: Optional[str]     # latest raw user answer

    # ── Intermediate ─────────────────────────────────────────────────────
    raw_offers: list[dict[str, Any]]     # normalized Duffel offer dicts
    ranked_flights: list[dict[str, Any]] # RankedFlight as dicts

    # ── Output ───────────────────────────────────────────────────────────
    recommended_flights: list[dict[str, Any]]
    candidates_found: int
    warnings: Annotated[list[str], operator.add]   # nodes append, never overwrite
    error: Optional[str]
