"""Map the input pipeline's OrchestratorInput straight onto a HotelSearchRequest.

Pure field mapping, no API calls. The trip has no country field, so
country_code is left empty; resolve_location_node fills it in from the same
city geocoding call that provides the coordinates.
"""

from __future__ import annotations

from .agent import HotelSearchRequest


def _preferences(ex) -> str:
    parts: list[str] = []
    if ex.num_travelers:
        parts.append(f"{ex.num_travelers} traveler(s)")
    if ex.traveler_details:
        parts.append(", ".join(ex.traveler_details))
    if ex.accommodation_type:
        parts.append(f"accommodation: {ex.accommodation_type}")
    if ex.budget:
        nights = ex.duration_max_days or ex.duration_min_days
        trip = f" for a {nights}-day trip" if nights else ""
        parts.append(f"total trip budget {ex.budget:,.0f} {ex.currency or ''}".rstrip() + trip)
    if ex.interests:
        parts.append("interests: " + ", ".join(ex.interests))
    return "; ".join(parts)


def hotel_request_from_trip(oi) -> HotelSearchRequest:
    """city <- destination, preferences <- traveller details, vibe <- trip_style."""

    ex = oi.trip_extraction
    city = (ex.destination or "").strip()
    if not city:
        raise ValueError("No destination was collected, so hotels cannot be searched.")

    return HotelSearchRequest(
        city=city,
        preferences=_preferences(ex),
        vibe=ex.trip_style or "",
    )
