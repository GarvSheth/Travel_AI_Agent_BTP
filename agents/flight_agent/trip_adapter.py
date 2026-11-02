"""Map the input pipeline's OrchestratorInput onto a FlightSearchRequest.

Pure field mapping, no API calls. Origin and destination come from the trip
extraction; if they are city names rather than IATA codes, the search node
will pass them through as-is (Duffel requires IATA codes, so full city-name
resolution is a follow-up concern).
"""
from __future__ import annotations

from .agent import FlightSearchRequest


def _preferences(ex) -> str:
    parts: list[str] = []
    if ex.num_travelers:
        parts.append(f"{ex.num_travelers} traveler(s)")
    if ex.traveler_details:
        parts.append(", ".join(ex.traveler_details))
    if ex.trip_style:
        parts.append(f"trip style: {ex.trip_style}")
    if ex.budget:
        nights = ex.duration_max_days or ex.duration_min_days
        trip = f" for a {nights}-day trip" if nights else ""
        parts.append(f"total budget {ex.budget:,.0f} {ex.currency or ''}".rstrip() + trip)
    if ex.interests:
        parts.append("interests: " + ", ".join(ex.interests))
    return "; ".join(parts)


def flight_request_from_trip(oi) -> FlightSearchRequest:
    """origin <- origin, destination <- destination, dates and pax from trip."""
    ex = oi.trip_extraction

    origin = (ex.origin or "").strip()
    if not origin:
        raise ValueError("No origin was collected, so flights cannot be searched.")

    destination = (ex.destination or "").strip()
    if not destination:
        raise ValueError("No destination was collected, so flights cannot be searched.")

    return FlightSearchRequest(
        origin=origin,
        destination=destination,
        departure_date=ex.departure_date or "",
        return_date=ex.return_date or None,
        num_passengers=ex.num_travelers or 1,
        budget_usd=ex.budget or None,
        currency=ex.currency or "USD",
        preferences=_preferences(ex),
    )
