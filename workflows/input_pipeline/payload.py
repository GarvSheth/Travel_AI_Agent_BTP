"""Final orchestrator payload and handoff to the main LangGraph state."""
from __future__ import annotations

from models.trip import TripRequest
from schemas.trip_schema import OrchestratorInput, TripExtraction
from workflows.input_pipeline.orchestrator_prompt import build_orchestrator_prompt

DEFAULT_TASKS = ["transport", "accommodation", "experiences", "itinerary"]


def build_orchestrator_input(original_prompt: str, trip: TripExtraction) -> OrchestratorInput:
    tasks = trip.requested_tasks or list(DEFAULT_TASKS)
    oi = OrchestratorInput(
        original_prompt=original_prompt,
        trip_extraction=trip,
        missing_information=[],
        clarification_required=False,
        requested_tasks=tasks,
    )
    _trip_request(oi)  # fail fast if the payload doesn't satisfy the orchestrator contract
    oi.orchestrator_prompt = build_orchestrator_prompt(original_prompt, trip, tasks)
    return oi


def _trip_request(oi: OrchestratorInput) -> dict:
    ex = oi.trip_extraction
    trip_request = {
        "destination": ex.destination,
        "origin": ex.origin,
        "departure_date": ex.departure_date,
        "return_date": ex.return_date,
        "duration_min_days": ex.duration_min_days,
        "duration_max_days": ex.duration_max_days,
        "num_travelers": ex.num_travelers,
        "budget": ex.budget,
        "budget_usd": ex.budget if ex.currency == "USD" else None,
        "currency": ex.currency,
        "preferred_currency": ex.currency,
        "accommodation_type": ex.accommodation_type or "any",
        "trip_style": ex.trip_style,
        "interests": ex.interests,
        "traveler_details": ex.traveler_details,
    }
    TripRequest.model_validate(trip_request)
    return trip_request


def orchestrator_input_to_state(oi: OrchestratorInput) -> dict:
    """Initial main-graph state for a completed, validated OrchestratorInput."""
    return {
        "original_prompt": oi.original_prompt,
        "user_context": oi.user_context,
        "requested_tasks": oi.requested_tasks,
        "missing_information": oi.missing_information,
        "clarification_required": oi.clarification_required,
        "trip_request": _trip_request(oi),
        "orchestrator_prompt": oi.orchestrator_prompt,
        "phase": "itinerary",
        "messages": [],
    }
