"""Render the validated trip payload into the booking orchestrator's prompt.

Every value comes from the final TripExtraction; nothing is inferred or invented.
"""
from __future__ import annotations

from datetime import datetime

from schemas.trip_schema import TripExtraction

_NO_PREF = "No preference stated"


def _day(d: str) -> str:
    return f"{d} ({datetime.strptime(d, '%Y-%m-%d').strftime('%A')})"


def _money(amount: float, currency: str) -> str:
    return f"{amount:,.0f} {currency}"


def _transport_section(n: int, ex: TripExtraction) -> str:
    return f"""### {n}. Transport (transport agent)
- Find OUTBOUND options from {ex.origin} to {ex.destination} on {ex.departure_date} and RETURN options from {ex.destination} to {ex.origin} on {ex.return_date}, for {ex.num_travelers} traveler(s).
- Consider every practical mode (flight, train, bus). Include connecting routes when there is no direct service, and name the connection point.
- For each option report: mode, operator/carrier, departure and arrival times, total journey time, number of stops, price per person and total price in {ex.currency}.
- Shortlist the 3 best options per direction, balancing price, total travel time and convenience."""


def _accommodation_section(n: int, ex: TripExtraction, nights: int) -> str:
    stay_type = ex.accommodation_type if ex.accommodation_type not in (None, "any") else None
    location_hint = (
        f"Prefer locations that suit a {ex.trip_style} trip"
        + (f" and are close to {', '.join(ex.interests)} attractions" if ex.interests else "")
        + "."
        if ex.trip_style or ex.interests else
        "Prefer central, well-connected locations."
    )
    return f"""### {n}. Accommodation (accommodation agent)
- Find {stay_type or "accommodation (any suitable type)"} options in {ex.destination}: check-in {ex.departure_date}, check-out {ex.return_date} ({nights} night(s)), for {ex.num_travelers} guest(s).
- {location_hint}
- For each option report: name, area/neighbourhood, property type, star rating, guest rating and review count, price per night, total stay price in {ex.currency}, and cancellation policy.
- Shortlist the 3 best options."""


def build_orchestrator_prompt(
    original_prompt: str, trip: TripExtraction, requested_tasks: list[str]
) -> str:
    ex = trip
    dep = datetime.strptime(ex.departure_date, "%Y-%m-%d").date()
    ret = datetime.strptime(ex.return_date, "%Y-%m-%d").date()
    nights = max((ret - dep).days, 1)
    days = (ret - dep).days + 1
    budget = _money(ex.budget, ex.currency)
    per_person = _money(ex.budget / ex.num_travelers, ex.currency)
    party = f" ({', '.join(ex.traveler_details)})" if ex.traveler_details else ""

    tasks: list[str] = []
    if "transport" in requested_tasks:
        tasks.append(_transport_section(len(tasks) + 1, ex))
    if "accommodation" in requested_tasks:
        tasks.append(_accommodation_section(len(tasks) + 1, ex, nights))
    tasks_text = "\n\n".join(tasks) or "No transport or accommodation tasks were requested."

    return f"""You are the booking orchestrator of an AI travel agent. Coordinate the specialist agents to find and shortlist transport and accommodation for the trip below. Use ONLY the details given here — do not invent, assume or change any of them.

## Trip summary
- Origin: {ex.origin}
- Destination: {ex.destination}
- Departure date: {_day(ex.departure_date)}
- Return date: {_day(ex.return_date)}
- Duration: {days} day(s) / {nights} night(s)
- Travelers: {ex.num_travelers}{party}
- Total trip budget: {budget} (≈ {per_person} per traveler)

## Traveler preferences
- Trip style: {ex.trip_style or _NO_PREF}
- Accommodation type: {ex.accommodation_type or _NO_PREF}
- Interests: {", ".join(ex.interests) if ex.interests else _NO_PREF}

## Tasks
{tasks_text}

## Constraints
- The total budget of {budget} covers the whole trip. Transport plus accommodation must fit within it and leave room for food and activities.
- Quote every price in {ex.currency}. If a source returns another currency, show the converted {ex.currency} amount alongside the original.
- Do NOT make any booking or payment. Present the shortlists for the traveler's approval first.
- If something is unavailable (no route, no rooms, sold out), say so explicitly. Never substitute different dates, places or traveler counts.

## Original traveler request
"{original_prompt}"

## Response format
Return JSON only:
{{
  "transport": {{"outbound": [...], "return": [...]}},
  "accommodation": [...],
  "estimated_total": {{"amount": <number>, "currency": "{ex.currency}"}},
  "within_budget": <true|false>,
  "notes": ["<anything the traveler should know>"]
}}"""
