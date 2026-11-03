"""LangGraph nodes for the hotel recommendation workflow.

    clarify_accommodation_location -> (collect_location_answer -> clarify ...)
    -> clarify_star_rating -> resolve_location -> search_hotels -> load_facilities -> rank_hotels
    -> get_hotel_details -> prepare_output

Nodes return only the keys they change. ``warnings`` uses an additive reducer,
so each node returns just its new warnings. Failures that make the whole search
impossible set ``error`` and the router sends the graph to ``prepare_output``.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from langgraph.types import interrupt

from config.settings import (
    HOTEL_LOCATION_MAX_KM_FROM_CITY,
    HOTEL_MAX_LOCATION_CLARIFICATIONS,
    HOTEL_SEARCH_LIMIT,
    HOTEL_TOP_N,
)
from services.geocoding import GeocodingError, GeoLocation, geocode
from services.hotel_provider.nuitee_client import NuiteeAPIError, NuiteeAuthError, NuiteeClient

from .agent import HotelAgent, RecommendedHotel
from .clarification import decide_accommodation_location, parse_star_answer
from .ranking import HotelRankingError, distance_km, haversine_km, rank_hotels
from .state import HotelState

logger = logging.getLogger(__name__)


def _valid(loc: GeoLocation | None) -> bool:
    """Real coordinates only: finite, in range and not the (0, 0) placeholder."""
    if loc is None:
        return False
    lat, lon = loc.latitude, loc.longitude
    return (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        and math.isfinite(lat) and math.isfinite(lon)
        and -90 <= lat <= 90 and -180 <= lon <= 180
        and not (lat == 0 and lon == 0)
    )


def _safe_geocode(query: str, country_code: str | None) -> GeoLocation | None:
    try:
        loc = geocode(query, country_code)
    except GeocodingError as exc:
        logger.warning("%s", exc)
        return None
    return loc if _valid(loc) else None


class HotelWorkflowNodes:
    """Node implementations bound to one Nuitee client and LLM (injectable for tests)."""

    def __init__(self, client: NuiteeClient | None = None, llm=None) -> None:
        self._client = client
        self._llm = llm             # None -> services.llm_client.get_client() when needed

    @property
    def client(self) -> NuiteeClient:
        if self._client is None:
            self._client = NuiteeClient()      # raises NuiteeAuthError without a key
        return self._client

    # ── 1. Clarification ────────────────────────────────────────────────

    def clarify_accommodation_location_node(self, state: HotelState) -> dict:
        """Decide whether the stay location is clear; if not, generate a question."""
        city = state["city"]
        given = (state.get("location") or "").strip()
        answers = state.get("clarification_history") or []

        # Location supplied up front and nothing asked yet: no LLM call needed.
        if given and not answers:
            logger.info("Accommodation location given up front: %r", given)
            return {"clarification_required": False, "clarification_question": None,
                    "accommodation_location": given, "location_kind": "specific"}

        try:
            decision = decide_accommodation_location(
                city=city, preferences=state.get("preferences", ""),
                vibe=state.get("vibe", ""), answers=answers, llm=self._llm,
            )
        except Exception as exc:  # LLM unavailable / bad output: don't block the search
            logger.warning("Location clarification failed: %s", exc)
            return {"clarification_required": False, "clarification_question": None,
                    "accommodation_location": None, "location_kind": "city_centre",
                    "warnings": [f"Couldn't work out a preferred area; searching around "
                                 f"central {city}."]}

        if decision.status == "unclear":
            if state.get("clarification_attempts", 0) >= HOTEL_MAX_LOCATION_CLARIFICATIONS:
                logger.info("Location still unclear after %d answers; using city.",
                            state.get("clarification_attempts", 0))
                return {"clarification_required": False, "clarification_question": None,
                        "accommodation_location": None, "location_kind": "city_centre",
                        "warnings": [f"Couldn't pin down a preferred area; searching "
                                     f"around central {city}."]}
            return {"clarification_required": True,
                    "clarification_question": decision.question}

        update = {"clarification_required": False, "clarification_question": None,
                  "accommodation_location": decision.location_query or None,
                  "location_kind": decision.status,
                  "star_question": decision.star_question or None}
        if decision.stars_mentioned and state.get("star_rating") is None:
            logger.info("Star category already stated: %s", decision.stars_mentioned)
            update["star_rating"] = decision.stars_mentioned
        return update

    def collect_location_answer_node(self, state: HotelState) -> dict:
        """Pause the graph until the user answers (resumed with Command(resume=answer)).

        Kept separate from the LLM node because LangGraph re-runs an interrupted
        node from the top on resume; here that only re-reads the question.
        """
        question = state["clarification_question"]
        answer = str(interrupt({"question": question}) or "").strip()
        logger.info("Accommodation location answer: %r", answer)
        history = list(state.get("clarification_history") or [])
        history.append({"question": question, "answer": answer or "(no answer)"})
        return {"user_location_answer": answer,
                "clarification_history": history,
                "clarification_attempts": state.get("clarification_attempts", 0) + 1}

    def clarify_star_rating_node(self, state: HotelState) -> dict:
        """Ask which star category the user wants, unless it is already known.

        The interrupt comes first, so on resume the node re-runs straight to
        the answer and the LLM parses it exactly once.
        """
        if state.get("star_rating") is not None:
            return {}
        city = state["city"]
        question = state.get("star_question") or (
            f"Which hotel star category would you like in {city}: 3-star, 4-star, "
            "5-star, or no preference?")
        answer = str(interrupt({"question": question}) or "").strip()
        logger.info("Star category answer: %r", answer)
        try:
            stars = parse_star_answer(answer, llm=self._llm)
        except Exception as exc:  # unparseable answer: search without a star filter
            logger.warning("Star answer not understood: %s", exc)
            return {"star_rating": [], "star_answer": answer,
                    "warnings": ["Couldn't understand the star preference; showing all "
                                 "star categories."]}
        return {"star_rating": stars, "star_answer": answer}

    # ── 2. Coordinates ──────────────────────────────────────────────────

    def resolve_location_node(self, state: HotelState) -> dict:
        """Geocode the chosen place; fall back to the city's own coordinates."""
        city, cc = state["city"], state.get("country_code") or ""

        city_loc = _safe_geocode(city, cc or None)
        if city_loc is None:
            logger.error("City-level geocoding failed for %s, %s.", city, cc or "?")
            return {"error": f"Could not find coordinates for {city}, so hotels "
                             "cannot be searched. Check the city name."}
        if not cc:                       # trip input has no country: take it from the geocoder
            cc = city_loc.country_code or ""
            if not cc:
                return {"error": f"Could not determine which country {city} is in."}
            logger.info("Country for %s resolved to %s.", city, cc)

        place = state.get("accommodation_location")
        warnings: list[str] = []
        if state.get("location_kind") == "specific" and place:
            query = place if city.lower() in place.lower() else f"{place}, {city}"
            loc = _safe_geocode(query, cc)
            if loc is not None:
                km = haversine_km(city_loc.latitude, city_loc.longitude,
                                  loc.latitude, loc.longitude)
                if km <= HOTEL_LOCATION_MAX_KM_FROM_CITY:
                    logger.info("Resolved %r -> (%.5f, %.5f)", query, loc.latitude, loc.longitude)
                    return {"latitude": loc.latitude, "longitude": loc.longitude,
                            "location_label": loc.display_name, "location_source": "landmark",
                            "country_code": cc}
                logger.warning("%r resolved %.0f km from %s; treating as ambiguous.",
                               query, km, city)
            warnings.append(f"Could not locate '{place}' in {city}; searching around "
                            f"central {city} instead.")

        source = "city_centre" if state.get("location_kind") == "city_centre" else "city_fallback"
        return {"latitude": city_loc.latitude, "longitude": city_loc.longitude,
                "location_label": city_loc.display_name, "location_source": source,
                "country_code": cc, "warnings": warnings}

    # ── 3. Search ───────────────────────────────────────────────────────

    def search_hotels_node(self, state: HotelState) -> dict:
        try:
            hotels = self.client.search_hotels(
                country_code=state["country_code"],
                city_name=state["city"],
                latitude=state["latitude"],
                longitude=state["longitude"],
                radius=state["radius"],
                star_rating=state.get("star_rating") or None,
                min_rating=state.get("min_rating"),
                limit=HOTEL_SEARCH_LIMIT,
            )
        except NuiteeAuthError:
            return {"error": "Hotel API key missing. Set NUITEE_API_KEY in .env."}
        except (NuiteeAPIError, ValueError) as exc:
            return {"error": f"Hotel search failed: {exc}"}
        return {"hotels": hotels, "candidates_found": len(hotels)}

    def load_facilities_node(self, state: HotelState) -> dict:
        try:
            return {"facilities": self.client.get_facilities()}
        except NuiteeAPIError as exc:
            logger.warning("Facility lookup failed: %s", exc)
            return {"facilities": None,
                    "warnings": ["Facility names unavailable; facilities treated as unknown."]}

    # ── 4. Rank ─────────────────────────────────────────────────────────

    def rank_hotels_node(self, state: HotelState) -> dict:
        try:
            ranked = rank_hotels(
                state["hotels"],
                preferences=state.get("preferences", ""),
                mandatory_requirements=state.get("mandatory_requirements") or [],
                vibe=state.get("vibe", ""),
                facilities=state.get("facilities"),
                reference=(state["latitude"], state["longitude"]),
                top_n=HOTEL_TOP_N,
                llm=self._llm,
                star_filter=state.get("star_rating") or None,
            )
        except HotelRankingError as exc:
            return {"error": f"Could not rank hotels: {exc}"}
        except RuntimeError as exc:  # LLM client not initialised / API key problems
            return {"error": str(exc)}
        return {"ranked_hotels": [asdict(r) for r in ranked]}

    # ── 5. Describe ─────────────────────────────────────────────────────

    def get_hotel_details_node(self, state: HotelState) -> dict:
        """Build the top hotels and fetch their Ask-API descriptions concurrently."""
        by_id = {str(h["id"]): h for h in state["hotels"]}
        reference = (state["latitude"], state["longitude"])
        hotels = [_build(r, by_id[r["hotel_id"]], reference) for r in state["ranked_hotels"]]
        agent = HotelAgent(self.client)    # asks HOTEL_DESCRIPTION_QUESTION via /data/hotel/ask

        def fetch(hotel: RecommendedHotel) -> None:
            try:
                hotel.description = agent.get_hotel_details(hotel.hotel_id)
            except (NuiteeAPIError, ValueError) as exc:
                logger.warning("Description failed for %s: %s", hotel.hotel_id, exc)
                hotel.description_error = "Description unavailable for this hotel."

        if hotels:
            with ThreadPoolExecutor(max_workers=len(hotels)) as pool:
                list(pool.map(fetch, hotels))
        return {"recommended_hotels": [asdict(h) for h in hotels]}

    # ── 6. Output ───────────────────────────────────────────────────────

    def prepare_output_node(self, state: HotelState) -> dict:
        out = {
            "recommended_hotels": state.get("recommended_hotels") or [],
            "candidates_found": state.get("candidates_found", 0),
        }
        if not state.get("error"):
            # The only terminal output of the workflows: printed once the hotels are ready.
            print(f"Hotels fetched: {out['candidates_found']}", flush=True)
        return out


def _build(ranked: dict, hotel: dict[str, Any], reference: tuple[float, float]) -> RecommendedHotel:
    return RecommendedHotel(
        hotel_id=ranked["hotel_id"],
        name=ranked["hotel_name"],
        rank=ranked["rank"],
        match_score=ranked["match_score"],
        explanation=ranked["explanation"],
        stars=hotel.get("stars"),
        guest_rating=hotel.get("rating"),
        review_count=hotel.get("reviewCount"),
        address=hotel.get("address") or None,
        city=hotel.get("city") or None,
        image_url=hotel.get("main_photo") or hotel.get("thumbnail") or None,
        distance_km=distance_km(hotel, reference),
    )
