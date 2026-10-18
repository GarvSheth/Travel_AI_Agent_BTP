"""LangGraph state for the hotel recommendation workflow.

Everything stored here is plain data (str / float / list / dict) so the
checkpointer can persist it across the clarification interrupt.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class HotelState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────
    city: str
    country_code: str
    location: Optional[str]                 # landmark / area given up front, if any
    preferences: str
    mandatory_requirements: list[str]
    vibe: str
    star_rating: Optional[list[float]]      # None = not asked yet; [] = no preference
    min_rating: Optional[float]
    radius: int

    # ── Clarification ────────────────────────────────────────────────────
    clarification_question: Optional[str]
    clarification_required: bool
    clarification_attempts: int             # questions asked so far
    clarification_history: list[dict]       # [{question, answer}] shown to the LLM
    user_location_answer: Optional[str]     # latest raw answer from the user
    accommodation_location: Optional[str]   # geocodable place, e.g. "IIM Ahmedabad"
    location_kind: Optional[str]            # "specific" | "city_centre"
    star_question: Optional[str]            # LLM-written star-category question
    star_answer: Optional[str]              # user's raw answer to it

    # ── Location (always from geocoding, never from the LLM) ─────────────
    latitude: Optional[float]
    longitude: Optional[float]
    location_label: Optional[str]
    location_source: Optional[str]          # "landmark" | "city_centre" | "city_fallback"

    # ── Intermediate ─────────────────────────────────────────────────────
    hotels: list[dict[str, Any]]            # raw /data/hotels records
    facilities: Optional[dict[int, str]]
    ranked_hotels: list[dict[str, Any]]     # RankedHotel as dicts

    # ── Output ───────────────────────────────────────────────────────────
    recommended_hotels: list[dict[str, Any]]  # RecommendedHotel as dicts
    candidates_found: int
    warnings: Annotated[list[str], operator.add]   # nodes append, never overwrite
    error: Optional[str]
