"""Pydantic schemas for trip extraction and orchestrator input.

TripExtraction  — raw fields extracted from the user's natural-language prompt.
OrchestratorInput — validated envelope passed to the LangGraph orchestration layer.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TripExtraction(BaseModel):
    """Trip details extracted from the user's natural-language prompt."""
    destination: Optional[str] = None
    origin: Optional[str] = None
    departure_date: Optional[str] = None        # YYYY-MM-DD
    return_date: Optional[str] = None           # YYYY-MM-DD
    duration_min_days: Optional[int] = None     # lower bound when user says "3-4 days"
    duration_max_days: Optional[int] = None
    num_travelers: Optional[int] = None
    budget: Optional[float] = None              # amount in the stated currency
    currency: Optional[str] = None              # None = not stated (never default to USD)
    trip_style: Optional[str] = None
    accommodation_type: Optional[str] = None
    interests: list[str] = Field(default_factory=list)
    traveler_details: list[str] = Field(default_factory=list)  # e.g. ["couple", "with kids"]
    requested_tasks: list[str] = Field(default_factory=list)

    @field_validator("num_travelers")
    @classmethod
    def check_travelers(cls, v):
        if v is not None and not (1 <= v <= 20):
            raise ValueError(f"num_travelers must be 1–20, got {v}")
        return v

    @field_validator("budget")
    @classmethod
    def check_budget(cls, v):
        if v is not None and v <= 0:
            raise ValueError(f"budget must be positive, got {v}")
        return v

    @field_validator("departure_date", "return_date")
    @classmethod
    def check_date(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD, got '{v}'")
        return v


class OrchestratorInput(BaseModel):
    """Validated input envelope passed to the LangGraph orchestration layer."""
    invocation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    original_prompt: str
    user_context: dict = Field(default_factory=dict)   # stub until DB layer is ready
    trip_extraction: TripExtraction
    missing_information: list[str] = Field(default_factory=list)
    clarification_required: bool = False
    requested_tasks: list[str] = Field(default_factory=list)
    orchestrator_prompt: str = ""                       # rendered from trip_extraction
