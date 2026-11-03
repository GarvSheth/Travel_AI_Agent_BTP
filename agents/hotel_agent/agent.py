"""Hotel Agent data types, plus live room prices for a selected hotel.

The recommendation workflow itself (clarify location -> geocode -> search ->
facilities -> LLM rank -> Ask-API descriptions) is a LangGraph graph: see
``graph.py`` / ``nodes.py`` and the public ``search_hotels(request)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from config.settings import (
    HOTEL_DESCRIPTION_QUESTION,
    HOTEL_GUEST_NATIONALITY,
    HOTEL_MIN_RATING,
    HOTEL_PAYMENT_METHOD,
    HOTEL_RATES_MAX_OFFERS,
    HOTEL_SEARCH_RADIUS_M,
)
from services.hotel_provider.nuitee_client import NuiteeClient

from .booking import (
    BookingResult,
    GuestDetails,
    PrebookResult,
    booking_guests,
    parse_booking,
    parse_prebook,
)

logger = logging.getLogger(__name__)


def rooms_for_adults(adults: int) -> int:
    """rooms = [n/2] (greatest integer of n/2), never fewer than one room."""
    return max(1, int(adults) // 2)


@dataclass
class StayDetails:
    """Dates and guests used to price rooms (from the trip, or the Hotel Finder form)."""
    check_in: str                           # YYYY-MM-DD
    check_out: str                          # YYYY-MM-DD
    adults: int = 2
    rooms: int = 1
    currency: str = "INR"

    @property
    def nights(self) -> int:
        return (date.fromisoformat(self.check_out) - date.fromisoformat(self.check_in)).days


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
    adults: int | None = None               # travellers; stored in graph state as "adults"
    stay: StayDetails | None = None         # dates + currency, to price a selected hotel


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
    stay: StayDetails | None = None


@dataclass
class RoomOffer:
    offer_id: str                           # needed for /rates/prebook
    room_name: str
    board: str | None                       # e.g. "Room Only", "Breakfast Included"
    refundable: bool | None                 # None = unknown
    max_occupancy: int | None
    total: float                            # whole stay, all rooms
    currency: str
    taxes_included: bool | None


@dataclass
class HotelRates:
    hotel_id: str
    stay: StayDetails
    offers: list[RoomOffer]                 # cheapest first; [] = no availability
    sandbox: bool = False                   # True -> test prices from a sandbox key


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

    def get_room_offers(self, hotel_id: str, stay: StayDetails) -> HotelRates:
        """Live room prices for one hotel via POST /hotels/rates (hotelIds)."""
        response = self._client.search_hotel_rates(
            hotel_ids=[hotel_id],
            check_in=stay.check_in,
            check_out=stay.check_out,
            adults=stay.adults,
            rooms=stay.rooms,
            currency=stay.currency,
            guest_nationality=HOTEL_GUEST_NATIONALITY,
        )
        return HotelRates(
            hotel_id=hotel_id,
            stay=stay,
            offers=parse_room_offers(response, hotel_id),
            sandbox=bool(response.get("sandbox")),
        )

    # ── Booking ──────────────────────────────────────────────────────────

    def prebook(self, offer_id: str, hotel_name: str) -> PrebookResult:
        """Re-check the chosen room offer and hold it for booking."""
        return parse_prebook(self._client.prebook_rate(offer_id), hotel_name)

    def book(self, prebook: PrebookResult, guest: GuestDetails) -> BookingResult:
        """Final booking. Pays with HOTEL_PAYMENT_METHOD (simulated on a sandbox key)."""
        response = self._client.book_rate(
            prebook.prebook_id,
            holder=guest.as_holder(),
            guests=booking_guests(guest, prebook.rooms),
            payment_method=HOTEL_PAYMENT_METHOD,
            client_reference=f"travel-ai-{prebook.prebook_id}",
        )
        return parse_booking(response, prebook.hotel_name)


def _refundable(tag: Any) -> bool | None:
    return {"RFN": True, "NRFN": False}.get(str(tag or "").upper())


def parse_room_offers(
    response: dict[str, Any], hotel_id: str, limit: int = HOTEL_RATES_MAX_OFFERS
) -> list[RoomOffer]:
    """Cheapest offer per (room, board, refundable), sorted by price, at most ``limit``.

    Shape: data[] -> {hotelId, roomTypes[] -> {offerId, offerRetailRate{amount, currency},
    rates[0]{name, boardName, maxOccupancy, cancellationPolicies.refundableTag,
    retailRate.taxesAndFees[]}}}. Offers without a price or offerId are skipped.
    """
    data = response.get("data")
    hotels = [h for h in data if isinstance(h, dict)] if isinstance(data, list) else []
    hotel = next((h for h in hotels if h.get("hotelId") == hotel_id), None)
    if hotel is None:
        return []

    best: dict[tuple, RoomOffer] = {}
    for room_type in hotel.get("roomTypes") or []:
        if not isinstance(room_type, dict):
            continue
        price = room_type.get("offerRetailRate") or {}
        amount = price.get("amount")
        offer_id = room_type.get("offerId")
        if not isinstance(amount, (int, float)) or not offer_id:
            continue
        rates = room_type.get("rates") or [{}]
        rate = rates[0] if isinstance(rates[0], dict) else {}
        taxes = (rate.get("retailRate") or {}).get("taxesAndFees")
        included = None
        if isinstance(taxes, list) and taxes:
            included = all(t.get("included") for t in taxes if isinstance(t, dict))
        offer = RoomOffer(
            offer_id=str(offer_id),
            room_name=str(rate.get("name") or "Room"),
            board=rate.get("boardName") or None,
            refundable=_refundable((rate.get("cancellationPolicies") or {}).get("refundableTag")),
            max_occupancy=rate.get("maxOccupancy"),
            total=float(amount),
            currency=str(price.get("currency") or ""),
            taxes_included=included,
        )
        key = (offer.room_name, offer.board, offer.refundable)
        if key not in best or offer.total < best[key].total:
            best[key] = offer

    return sorted(best.values(), key=lambda o: o.total)[:limit]
