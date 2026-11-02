"""LangGraph nodes for the flight recommendation workflow.

    clarify_flight_preferences → (collect_preference_answer → clarify ...)
    → search_flights → rank_flights → prepare_output

Nodes return only the keys they change. ``warnings`` uses an additive reducer,
so each node returns just its new warnings. Failures that make the whole search
impossible set ``error`` and the router sends the graph to ``prepare_output``.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

from langgraph.types import interrupt

from config.settings import (
    FLIGHT_MAX_CLARIFICATION_ATTEMPTS,
    FLIGHT_SEARCH_LIMIT,
    FLIGHT_TOP_N,
)
from services.flight_provider.duffel_client import (
    DuffelAPIError,
    DuffelAuthError,
    DuffelClient,
    normalize_offer,
)

from .agent import RecommendedFlight
from .clarification import decide_flight_preferences
from .ranking import FlightRankingError, RankedFlight, rank_flights
from .state import FlightState

logger = logging.getLogger(__name__)


class FlightWorkflowNodes:
    """Node implementations bound to one DuffelClient and LLM (injectable for tests)."""

    def __init__(self, client: DuffelClient | None = None, llm=None) -> None:
        self._client = client
        self._llm = llm

    @property
    def client(self) -> DuffelClient:
        if self._client is None:
            self._client = DuffelClient()
        return self._client

    # ── 1. Clarification ─────────────────────────────────────────────────

    def clarify_flight_preferences_node(self, state: FlightState) -> dict:
        """Decide whether flight preferences are clear; generate a question if not."""
        # Already have preferences from a prior clarification round or input
        if state.get("cabin_class") is not None and state.get("max_stops") is not None:
            return {"clarification_required": False, "clarification_question": None}

        origin = state.get("origin") or ""
        destination = state.get("destination") or ""
        answers = state.get("clarification_history") or []

        try:
            decision = decide_flight_preferences(
                origin=origin,
                destination=destination,
                preferences=state.get("preferences", ""),
                answers=answers,
                llm=self._llm,
            )
        except Exception as exc:
            logger.warning("Flight preference clarification failed: %s", exc)
            return {
                "clarification_required": False,
                "clarification_question": None,
                "cabin_class": "economy",
                "max_stops": 2,
                "preferred_departure_period": "any",
                "warnings": ["Couldn't determine preferences; searching economy, any stops."],
            }

        if decision.status == "unclear":
            if state.get("clarification_attempts", 0) >= FLIGHT_MAX_CLARIFICATION_ATTEMPTS:
                logger.info(
                    "Preferences still unclear after %d answers; using defaults.",
                    state.get("clarification_attempts", 0),
                )
                return {
                    "clarification_required": False,
                    "clarification_question": None,
                    "cabin_class": "economy",
                    "max_stops": 2,
                    "preferred_departure_period": "any",
                    "warnings": ["Couldn't determine preferences; searching economy, any stops."],
                }
            return {
                "clarification_required": True,
                "clarification_question": decision.question,
            }

        return {
            "clarification_required": False,
            "clarification_question": None,
            "cabin_class": decision.cabin_class,
            "max_stops": decision.max_stops,
            "preferred_departure_period": decision.preferred_departure_period,
        }

    def collect_preference_answer_node(self, state: FlightState) -> dict:
        """Pause the graph until the user answers (resumed via Command(resume=answer)).

        Kept separate from the LLM node because LangGraph re-runs an interrupted
        node from the top on resume; here that only re-reads the question.
        """
        question = state["clarification_question"]
        answer = str(interrupt({"question": question}) or "").strip()
        logger.info("Flight preference answer: %r", answer)

        history = list(state.get("clarification_history") or [])
        history.append({"question": question, "answer": answer or "(no answer)"})
        return {
            "preference_answer": answer,
            "clarification_history": history,
            "clarification_attempts": state.get("clarification_attempts", 0) + 1,
        }

    # ── 2. Search ─────────────────────────────────────────────────────────

    def search_flights_node(self, state: FlightState) -> dict:
        """Call Duffel to search for flight offers."""
        origin = state.get("origin") or ""
        destination = state.get("destination") or ""
        departure_date = state.get("departure_date") or ""

        if not origin or not destination or not departure_date:
            return {"error": "Origin, destination, and departure date are required to search flights."}

        cabin = state.get("cabin_class") or "economy"
        max_stops = state.get("max_stops")
        max_conn = max_stops if max_stops is not None and max_stops < 2 else None

        try:
            raw = self.client.search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                num_passengers=state.get("num_passengers") or 1,
                cabin_class=cabin,
                max_connections=max_conn,
                limit=FLIGHT_SEARCH_LIMIT,
            )
        except DuffelAuthError:
            return {"error": "Duffel API key missing. Set DUFFEL_API_KEY in .env."}
        except DuffelAPIError as exc:
            return {"error": f"Flight search failed: {exc}"}

        offers = [normalize_offer(o) for o in raw]
        logger.info("Duffel search returned %d normalized offers.", len(offers))
        return {"raw_offers": offers, "candidates_found": len(offers)}

    # ── 3. Rank ──────────────────────────────────────────────────────────

    def rank_flights_node(self, state: FlightState) -> dict:
        """Use the LLM to filter and rank flight offers."""
        try:
            ranked = rank_flights(
                state.get("raw_offers") or [],
                preferences=state.get("preferences", ""),
                cabin_class=state.get("cabin_class"),
                max_stops=state.get("max_stops"),
                preferred_period=state.get("preferred_departure_period"),
                budget_usd=state.get("budget_usd"),
                top_n=FLIGHT_TOP_N,
                llm=self._llm,
            )
        except FlightRankingError as exc:
            return {"error": f"Could not rank flights: {exc}"}
        except RuntimeError as exc:
            return {"error": str(exc)}
        return {"ranked_flights": [asdict(r) for r in ranked]}

    # ── 4. Output ─────────────────────────────────────────────────────────

    def prepare_output_node(self, state: FlightState) -> dict:
        """Merge ranked data with original offer data into RecommendedFlight dicts."""
        ranked = state.get("ranked_flights") or []
        by_id = {o["option_id"]: o for o in (state.get("raw_offers") or [])}

        recommended: list[dict] = []
        for r in ranked:
            offer = by_id.get(r["flight_id"]) or {}
            recommended.append(
                asdict(RecommendedFlight(
                    flight_id=r["flight_id"],
                    airline=r["airline"],
                    flight_number=r["flight_number"],
                    origin=offer.get("origin") or "",
                    destination=offer.get("destination") or "",
                    departure_time=offer.get("departure_time") or "",
                    arrival_time=offer.get("arrival_time") or "",
                    duration=offer.get("duration") or "",
                    price_usd=offer.get("price_usd") or 0.0,
                    currency=offer.get("currency") or "USD",
                    stops=offer.get("stops") or 0,
                    departure_period=offer.get("departure_period") or "any",
                    cabin_class=offer.get("cabin_class") or "",
                    rank=r["rank"],
                    match_score=r["match_score"],
                    explanation=r["explanation"],
                    source=offer.get("source") or "duffel",
                ))
            )

        if not state.get("error"):
            print(f"Flights fetched: {state.get('candidates_found', 0)}", flush=True)

        return {
            "recommended_flights": recommended,
            "candidates_found": state.get("candidates_found", 0),
        }
