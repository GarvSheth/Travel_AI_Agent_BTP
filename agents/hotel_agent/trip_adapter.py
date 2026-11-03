"""Map the input pipeline's OrchestratorInput straight onto a HotelSearchRequest.

Pure field mapping, no API calls. The trip has no country field, so
country_code is left empty; resolve_location_node fills it in from the same
city geocoding call that provides the coordinates.
"""

from __future__ import annotations

from datetime import date, timedelta

from .agent import HotelSearchRequest, StayDetails, rooms_for_adults


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


def _stay(ex) -> StayDetails | None:
    """check-in = departure, check-out = return (at least 1 night); rooms = [adults/2]."""
    try:
        check_in = date.fromisoformat(ex.departure_date)
        check_out = date.fromisoformat(ex.return_date)
    except (TypeError, ValueError):
        return None
    if check_out <= check_in:
        check_out = check_in + timedelta(days=1)
    adults = max(1, int(ex.num_travelers or 1))
    return StayDetails(check_in.isoformat(), check_out.isoformat(), adults=adults,
                       rooms=rooms_for_adults(adults), currency=(ex.currency or "INR").upper())


def hotel_request_from_trip(oi) -> HotelSearchRequest:
    """city <- destination, preferences <- traveller details, vibe <- trip_style,
    stay <- dates / travellers / currency (for pricing a selected hotel)."""

    ex = oi.trip_extraction
    city = (ex.destination or "").strip()
    if not city:
        raise ValueError("No destination was collected, so hotels cannot be searched.")

    return HotelSearchRequest(
        city=city,
        preferences=_preferences(ex),
        vibe=ex.trip_style or "",
        adults=max(1, int(ex.num_travelers or 1)),
        stay=_stay(ex),
    )
