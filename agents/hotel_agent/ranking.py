"""LLM filtering and ranking of Nuitee hotel search results.

No API calls happen here: the caller passes in hotels already fetched from
Nuitee. Only facts present in those API objects are shown to the LLM, and the
LLM's output is validated against the input (unknown IDs dropped, names taken
from the API data, scores clamped) so it cannot introduce new hotels.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DESCRIPTION_EXCERPT_CHARS = 500


class HotelRankingError(Exception):
    """Raised when the LLM ranking call fails or returns unusable output."""


@dataclass
class RankedHotel:
    hotel_id: str
    hotel_name: str
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
                    "hotel_id": {"type": "string"},
                    "hotel_name": {"type": "string"},
                    "rank": {"type": "integer"},
                    "match_score": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["hotel_id", "hotel_name", "rank", "match_score", "explanation"],
            },
        },
    },
    "required": ["rankings"],
}

_SYSTEM = """You are a hotel recommendation assistant. You rank real hotels returned by a
hotel data API against a traveller's preferences.

You receive:
- user_preferences: the traveller's own words about what they want.
- mandatory_requirements: things the hotel MUST satisfy.
- preferred_vibe: the atmosphere the traveller wants.
- hotels: hotel records from the API. A null / missing field means UNKNOWN.

Rules:
1. Use ONLY the information in the hotel records. Never invent or assume facilities,
   ratings, locations, prices, atmosphere or any other attribute.
2. Mandatory requirements are hard constraints. If a hotel's record shows it does not
   meet one (e.g. its facilities list is known and lacks the required facility, or its
   stars are below a required minimum), EXCLUDE that hotel.
   If the needed data is unknown (null), the hotel may be included, but the explanation
   must say the requirement could not be verified, and it must earn no score for it.
3. Then weigh softer preferences: vibe, distance from the preferred location
   (distance_km), star rating, guest rating (0-10) and review count.
4. match_score is 0-100 and reflects only evidence present in the record. Hotels with
   little known information must not receive a high score.
5. Return at most {top_n} hotels, best first, rank starting at 1. If fewer hotels
   qualify, return only those; if none qualify, return an empty list.
6. explanation: 1-2 short sentences naming the specific facts from the record that
   make it a match (and any unverified mandatory requirement).
7. hotel_id and hotel_name must be copied exactly from the records.

Return JSON: {{"rankings": [{{"hotel_id", "hotel_name", "rank", "match_score", "explanation"}}]}}"""


# ── Building the LLM view of each hotel ──────────────────────────────────────

def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_km(hotel: dict[str, Any], reference: tuple[float, float] | None) -> float | None:
    lat, lon = hotel.get("latitude"), hotel.get("longitude")
    if reference is None or not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    return round(haversine_km(reference[0], reference[1], lat, lon), 2)


def summarize_hotel(
    hotel: dict[str, Any],
    facilities: dict[int, str] | None,
    reference: tuple[float, float] | None,
) -> dict[str, Any]:
    """Compact, fact-only view of a /data/hotels record for the LLM."""

    facility_names: list[str] | None = None
    ids = hotel.get("facilityIds")
    if facilities is not None and isinstance(ids, list):
        facility_names = sorted({facilities[i] for i in ids if i in facilities})

    description = hotel.get("hotelDescription")
    excerpt = _strip_html(description)[:DESCRIPTION_EXCERPT_CHARS] if isinstance(description, str) else None

    chain = hotel.get("chain")
    pet = (hotel.get("accessibilityAttributes") or {}).get("petFriendly")

    return {
        "hotel_id": hotel.get("id"),
        "hotel_name": hotel.get("name"),
        "stars": hotel.get("stars"),
        "guest_rating": hotel.get("rating"),
        "review_count": hotel.get("reviewCount"),
        "address": hotel.get("address"),
        "city": hotel.get("city"),
        "distance_km": distance_km(hotel, reference),
        "chain": chain if chain and chain != "Not Available" else None,
        "pet_friendly": pet or None,
        "facilities": facility_names,
        "description_excerpt": excerpt or None,
    }


# ── Ranking ──────────────────────────────────────────────────────────────────

def rank_hotels(
    hotels: list[dict[str, Any]],
    *,
    preferences: str = "",
    mandatory_requirements: list[str] | None = None,
    vibe: str = "",
    facilities: dict[int, str] | None = None,
    reference: tuple[float, float] | None = None,
    top_n: int = 5,
    llm=None,
) -> list[RankedHotel]:
    """Filter and rank hotels with one LLM call; return at most ``top_n``."""

    if not hotels:
        return []

    if llm is None:
        from services.llm_client import get_client
        llm = get_client()

    by_id = {str(h["id"]): h for h in hotels if h.get("id")}
    payload = {
        "user_preferences": preferences or None,
        "mandatory_requirements": mandatory_requirements or [],
        "preferred_vibe": vibe or None,
        "hotels": [summarize_hotel(h, facilities, reference) for h in by_id.values()],
    }

    logger.info("Ranking %d hotels with the LLM (top_n=%d).", len(by_id), top_n)

    try:
        raw = llm.structured_output(
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            system=_SYSTEM.format(top_n=top_n),
            schema=RANKING_SCHEMA,
        )
    except Exception as exc:
        raise HotelRankingError(f"LLM ranking call failed: {exc}") from exc

    try:
        rankings = json.loads(raw).get("rankings")
    except (ValueError, AttributeError) as exc:
        raise HotelRankingError("LLM returned invalid JSON for hotel ranking.") from exc

    if not isinstance(rankings, list):
        raise HotelRankingError("LLM ranking response has no 'rankings' list.")

    return _validate_rankings(rankings, by_id, top_n)


def _validate_rankings(
    rankings: list[Any],
    by_id: dict[str, dict[str, Any]],
    top_n: int,
) -> list[RankedHotel]:
    """Drop invented/duplicate hotels, clamp scores and renumber ranks."""

    cleaned: list[tuple[int, RankedHotel]] = []
    seen: set[str] = set()

    for position, item in enumerate(rankings):
        if not isinstance(item, dict):
            continue

        hotel_id = str(item.get("hotel_id") or "").strip()
        if hotel_id not in by_id:
            logger.warning("LLM returned unknown hotel id %r; dropped.", hotel_id)
            continue
        if hotel_id in seen:
            continue

        try:
            score = int(round(float(item.get("match_score"))))
        except (TypeError, ValueError):
            logger.warning("LLM returned no usable score for %s; dropped.", hotel_id)
            continue

        try:
            llm_rank = int(item.get("rank"))
        except (TypeError, ValueError):
            llm_rank = position + 1

        seen.add(hotel_id)
        cleaned.append((
            llm_rank,
            RankedHotel(
                hotel_id=hotel_id,
                hotel_name=str(by_id[hotel_id].get("name") or item.get("hotel_name") or "Unknown"),
                rank=0,
                match_score=max(0, min(100, score)),
                explanation=str(item.get("explanation") or "").strip(),
            ),
        ))

    cleaned.sort(key=lambda pair: (pair[0], -pair[1].match_score))
    result = [hotel for _, hotel in cleaned[:top_n]]
    for index, hotel in enumerate(result, start=1):
        hotel.rank = index
    return result
