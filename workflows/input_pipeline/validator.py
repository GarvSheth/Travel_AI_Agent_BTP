"""Deterministic validation of TripExtraction against the orchestrator contract.

Required fields mirror models.trip.TripRequest, which the itinerary and specialist
nodes construct from trip_request. This module reports facts only; it never
decides what to say to the user.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

from schemas.trip_schema import TripExtraction

REQUIRED_FIELDS = [
    "destination", "origin", "departure_date", "return_date",
    "num_travelers", "budget", "currency",
]
OPTIONAL_FIELDS = ["trip_style", "accommodation_type", "interests", "traveler_details"]


class ValidationResult(BaseModel):
    is_valid: bool
    missing_fields: list[str] = Field(default_factory=list)
    invalid_fields: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    unfilled_optional_fields: list[str] = Field(default_factory=list)


def _is_empty(value) -> bool:
    return value is None or value == "" or value == []


def validate_trip(
    ex: TripExtraction,
    rejected_updates: Optional[list[dict]] = None,
    today: Optional[date] = None,
) -> ValidationResult:
    today = today or date.today()
    missing = [f for f in REQUIRED_FIELDS if _is_empty(getattr(ex, f))]
    invalid: list[str] = []
    errors: list[str] = []

    def flag(field: str, message: str) -> None:
        if field not in invalid:
            invalid.append(field)
        errors.append(f"{field}: {message}")

    for r in rejected_updates or []:
        flag(r["field"], f"rejected value {r['value']!r} ({r['error']})")

    dep = datetime.strptime(ex.departure_date, "%Y-%m-%d").date() if ex.departure_date else None
    ret = datetime.strptime(ex.return_date, "%Y-%m-%d").date() if ex.return_date else None
    if dep and dep < today:
        flag("departure_date", f"{dep} is in the past (today is {today})")
    if dep and ret and ret < dep:
        flag("return_date", f"{ret} is before departure_date {dep}")

    dmin, dmax = ex.duration_min_days, ex.duration_max_days
    if dmin is not None and dmin < 1:
        flag("duration_min_days", f"must be at least 1 day, got {dmin}")
    if dmin is not None and dmax is not None and dmin > dmax:
        flag("duration_max_days", f"max {dmax} is less than min {dmin}")
    if dep and ret and dmin is not None and dmax is not None and ret >= dep:
        nights = (ret - dep).days
        # "4 days, 14th to 18th" is commonly meant as 4 nights, so accept either count.
        if not (dmin <= nights <= dmax or dmin <= nights + 1 <= dmax):
            flag("return_date", f"dates span {nights + 1} days / {nights} nights "
                                f"but stated duration is {dmin}-{dmax} days")

    if ex.currency is not None and (len(ex.currency) != 3 or not ex.currency.isalpha()):
        flag("currency", f"{ex.currency!r} is not a 3-letter ISO currency code")

    missing = [f for f in missing if f not in invalid]
    unfilled_optional = [f for f in OPTIONAL_FIELDS if _is_empty(getattr(ex, f))]
    return ValidationResult(
        is_valid=not missing and not invalid,
        missing_fields=missing,
        invalid_fields=invalid,
        errors=errors,
        unfilled_optional_fields=unfilled_optional,
    )
