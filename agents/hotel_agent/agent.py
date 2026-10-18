"""Hotel Agent data types and live-rates helper.

The recommendation workflow itself (clarify location -> geocode -> search ->
facilities -> LLM rank -> Ask-API descriptions) is a LangGraph graph: see
``graph.py`` / ``nodes.py`` and the public ``search_hotels(request)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config.settings import HOTEL_DESCRIPTION_QUESTION, HOTEL_MIN_RATING, HOTEL_SEARCH_RADIUS_M
from services.hotel_provider.nuitee_client import NuiteeClient

logger = logging.getLogger(__name__)


@dataclass
class HotelSearchRequest:
    city: str
    country_code: str = ""                  # ISO-2, e.g. "IN"; "" -> taken from city geocoding
    location: str | None = None             # landmark / area; None -> the graph asks the user
    preferences: str = ""                   # the user's own words
    mandatory_requirements: list[str] = field(default_factory=list)
    vibe: str = ""
    star_rating: list[float] | None = None  # e.g. [4.0, 5.0]; None -> the graph asks the user
    min_rating: float | None = HOTEL_MIN_RATING  # guest rating 0-10
    radius: int = HOTEL_SEARCH_RADIUS_M


@dataclass
class ResolvedLocation:
    latitude: float
    longitude: float
    label: str
    source: str                             # "landmark" | "city_centre" | "city_fallback"


@dataclass
class RecommendedHotel:
    hotel_id: str
    name: str
    rank: int
    match_score: int
    explanation: str
    stars: float | None
    guest_rating: float | None
    review_count: int | None
    address: str | None
    city: str | None
    image_url: str | None
    distance_km: float | None
    description: str | None = None          # from /data/hotel/ask
    description_error: str | None = None


@dataclass
class HotelRecommendationResult:
    hotels: list[RecommendedHotel]
    candidates_found: int
    location: ResolvedLocation | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class HotelResult:
    accommodation_id: str
    name: str
    rating: str | None
    review_score: float | None
    cheapest_rate: float | None
    currency: str | None
    raw: dict[str, Any] = field(repr=False)


class HotelAgent:
    """Direct Nuitee helpers outside the recommendation graph (descriptions, live rates)."""

    def __init__(self, client: NuiteeClient | None = None) -> None:
        self._client = client or NuiteeClient()

    def get_hotel_details(
        self, hotel_id: str, question: str = HOTEL_DESCRIPTION_QUESTION
    ) -> str:
        """Description of one hotel from Nuitee's /data/hotel/ask."""
        return self._client.get_hotel_details(hotel_id, question)

    # ── Rates (live availability) ────────────────────────────────────────

    def search_and_recommend_hotels(
        self,
        city: str,
        country_code: str,
        check_in: str,
        check_out: str,
        adults: int = 2,
        rooms: int = 1,
        currency: str = "INR",
        guest_nationality: str = "IN",
    ) -> list[HotelResult]:
        """Search hotels and their available rates."""

        response = self._client.search_hotel_rates(
            city=city,
            country_code=country_code,
            check_in=check_in,
            check_out=check_out,
            adults=adults,
            rooms=rooms,
            currency=currency,
            guest_nationality=guest_nationality,
        )

        return self._parse_results(response)

    @staticmethod
    def _parse_results(response: dict[str, Any]) -> list[HotelResult]:
        """Parse Nuitee response.

        Final field mapping should be confirmed against an actual
        successful Nuitee API response.
        """

        data = response.get("data", [])

        if isinstance(data, dict):
            data = data.get("hotels", [])

        if not isinstance(data, list):
            return []

        hotels: list[HotelResult] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            hotel = item.get("hotel", item)

            if not isinstance(hotel, dict):
                continue

            hotel_id = hotel.get("hotelId") or hotel.get("id", "")
            name = hotel.get("name", "Unknown")

            hotels.append(
                HotelResult(
                    accommodation_id=str(hotel_id),
                    name=name,
                    rating=hotel.get("starRating", hotel.get("rating")),
                    review_score=hotel.get("reviewScore"),
                    cheapest_rate=None,
                    currency=None,
                    raw=item,
                )
            )

        return hotels
