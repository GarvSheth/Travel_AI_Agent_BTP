"""LLM filtering and ranking of Duffel flight offers.

No API calls happen here: the caller passes in offers already fetched and
normalized from Duffel. Only facts present in those records are shown to the
LLM, and the LLM's output is validated against the input (unknown IDs dropped,
names taken from the data, scores clamped) so it cannot introduce new flights.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FlightRankingError(Exception):
    """Raised when the LLM ranking call fails or returns unusable output."""


@dataclass
class RankedFlight:
    flight_id: str
    airline: str
    flight_number: str
    rank: int
    match_score: int          # 0-100
    explanation: str


RANKING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "airline": {"type": "string"},
                    "flight_number": {"type": "string"},
                    "rank": {"type": "integer"},
                    "match_score": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "flight_id", "airline", "flight_number",
                    "rank", "match_score", "explanation",
                ],
            },
        },
    },
    "required": ["rankings"],
}

_SYSTEM = """You are a flight recommendation assistant. You rank real flight offers
returned by the Duffel API against a traveller's preferences.

You receive:
- user_preferences: the traveller's own words about what they want.
- cabin_class: the cabin the traveller requested (or null for no preference).
- max_stops: the maximum number of stops the traveller wants (0=direct, 1, 2=any; null=any).
- preferred_departure_period: "morning", "afternoon", "evening", or "any".
- budget_usd: the traveller's total trip budget in USD (or null if unknown).
- flights: flight offer records from the Duffel API. A null / missing field means UNKNOWN.

Rules:
1. Use ONLY the information in the flight records. Never invent or assume price,
   airline, duration, stops, schedule, or any other attribute.
2. Hard constraints (exclude flights that violate them):
   - If max_stops is 0: exclude any flight with stops > 0.
   - If max_stops is 1: exclude any flight with stops > 1.
   - If budget_usd is set and a flight's price clearly exceeds it, exclude it.
   Null data never triggers exclusion.
3. Weigh softer preferences: price vs budget, stop count, departure timing,
   total duration, cabin class match, airline preference (if stated).
4. match_score is 0-100 and reflects only evidence in the record. Flights with
   little data must not receive a high score.
5. Return at most {top_n} flights, best first, rank starting at 1. Return fewer
   if fewer qualify; return an empty list if none qualify.
6. explanation: 1-2 short sentences naming the specific facts from the record
   that make it a good or poor match.
7. flight_id, airline, and flight_number must be copied exactly from the records.

Return JSON: {{"rankings": [{{"flight_id", "airline", "flight_number", "rank", "match_score", "explanation"}}]}}"""


def _parse_duration(iso: str) -> str:
    """Convert ISO 8601 duration to human-readable, e.g. 'PT5H30M' → '5h 30m'."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso or "")
    if not m:
        return iso or "unknown"
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    return " ".join(parts) or iso


def summarize_flight(offer: dict[str, Any]) -> dict[str, Any]:
    """Compact, fact-only view of a normalized offer for the LLM."""
    return {
        "flight_id": offer.get("option_id"),
        "airline": offer.get("airline"),
        "flight_number": offer.get("flight_number"),
        "origin": offer.get("origin"),
        "destination": offer.get("destination"),
        "departure_time": offer.get("departure_time"),
        "arrival_time": offer.get("arrival_time"),
        "duration": _parse_duration(offer.get("duration") or ""),
        "price": offer.get("price_usd"),
        "currency": offer.get("currency"),
        "stops": offer.get("stops"),
        "departure_period": offer.get("departure_period"),
        "cabin_class": offer.get("cabin_class"),
    }


def rank_flights(
    offers: list[dict[str, Any]],
    *,
    preferences: str = "",
    cabin_class: Optional[str] = None,
    max_stops: Optional[int] = None,
    preferred_period: Optional[str] = None,
    budget_usd: Optional[float] = None,
    top_n: int = 5,
    llm=None,
) -> list[RankedFlight]:
    """Filter and rank flight offers with one LLM call; return at most ``top_n``."""
    if not offers:
        return []

    if llm is None:
        from services.llm_client import get_client
        llm = get_client()

    by_id = {str(o["option_id"]): o for o in offers if o.get("option_id")}
    payload = {
        "user_preferences": preferences or None,
        "cabin_class": cabin_class or None,
        "max_stops": max_stops,
        "preferred_departure_period": preferred_period or "any",
        "budget_usd": budget_usd,
        "flights": [summarize_flight(o) for o in by_id.values()],
    }

    logger.info("Ranking %d flight offers with the LLM (top_n=%d).", len(by_id), top_n)

    try:
        raw = llm.structured_output(
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            system=_SYSTEM.format(top_n=top_n),
            schema=RANKING_SCHEMA,
        )
    except Exception as exc:
        raise FlightRankingError(f"LLM ranking call failed: {exc}") from exc

    try:
        rankings = json.loads(raw).get("rankings")
    except (ValueError, AttributeError) as exc:
        raise FlightRankingError("LLM returned invalid JSON for flight ranking.") from exc

    if not isinstance(rankings, list):
        raise FlightRankingError("LLM ranking response has no 'rankings' list.")

    return _validate_rankings(rankings, by_id, top_n)


def _validate_rankings(
    rankings: list[Any],
    by_id: dict[str, dict[str, Any]],
    top_n: int,
) -> list[RankedFlight]:
    """Drop invented/duplicate flights, clamp scores, renumber ranks."""
    cleaned: list[tuple[int, RankedFlight]] = []
    seen: set[str] = set()

    for position, item in enumerate(rankings):
        if not isinstance(item, dict):
            continue

        flight_id = str(item.get("flight_id") or "").strip()
        if flight_id not in by_id:
            logger.warning("LLM returned unknown flight id %r; dropped.", flight_id)
            continue
        if flight_id in seen:
            continue

        try:
            score = int(round(float(item.get("match_score"))))
        except (TypeError, ValueError):
            logger.warning("LLM returned no usable score for %s; dropped.", flight_id)
            continue

        try:
            llm_rank = int(item.get("rank"))
        except (TypeError, ValueError):
            llm_rank = position + 1

        offer = by_id[flight_id]
        seen.add(flight_id)
        cleaned.append((
            llm_rank,
            RankedFlight(
                flight_id=flight_id,
                airline=str(offer.get("airline") or item.get("airline") or "Unknown"),
                flight_number=str(offer.get("flight_number") or item.get("flight_number") or ""),
                rank=0,
                match_score=max(0, min(100, score)),
                explanation=str(item.get("explanation") or "").strip(),
            ),
        ))

    cleaned.sort(key=lambda pair: (pair[0], -pair[1].match_score))
    result = [flight for _, flight in cleaned[:top_n]]
    for index, flight in enumerate(result, start=1):
        flight.rank = index
    return result
