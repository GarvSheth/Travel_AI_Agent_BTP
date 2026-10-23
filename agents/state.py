from typing import Optional
from typing_extensions import TypedDict


class TravelAgentState(TypedDict):
    # Anthropic-format message dicts; plain list so the SDK receives them as-is
    messages: list

    # Populated after two-phase preference gathering
    trip_request: Optional[dict]   # serialized TripRequest

    # Populated after itinerary generation
    itinerary: Optional[dict]      # serialized Itinerary

    # Booking loop
    bookable_items: list           # [{"type": "flight"|"hotel", "day_index": int, "data": dict}]
    current_booking_index: int
    search_results: dict           # str(idx) → list[ScoredOption dicts]; f"{idx}_selected" → chosen
    item_decision: str             # "book" | "skip" | "show_more"
    booking_records: list          # list of BookingRecord dicts

    # Flow control
    plan_approved: bool
    phase: str    # "gathering_core" | "gathering_preferences" | "itinerary" | "booking" | "done"
    error_message: Optional[str]
    provider: str  # "gemini" | "anthropic"
    profile_id: str
    trip_id: Optional[str]
    correlation_id: str


INITIAL_STATE: TravelAgentState = {
    "messages": [],
    "trip_request": None,
    "itinerary": None,
    "bookable_items": [],
    "current_booking_index": 0,
    "search_results": {},
    "item_decision": "",
    "booking_records": [],
    "plan_approved": False,
    "phase": "gathering_core",
    "error_message": None,
    "provider": "gemini",
    "profile_id": "",
    "trip_id": None,
    "correlation_id": "",
}
