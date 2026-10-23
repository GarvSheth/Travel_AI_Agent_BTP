import json
from datetime import datetime

from models.trip import Itinerary, TripRequest
from services.llm_client import get_client
from services.persistence import create_storage
from services.profile_service import audit

_SYSTEM = """You are an expert travel planner. Generate a detailed day-by-day itinerary in JSON.

Rules:
- Include a flight segment on the first day (departure) and last day (return)
- Include a hotel stay for each night
- Activities should be realistic, varied, and respect the budget
- estimated_total_cost_usd should be a realistic estimate given the budget
- All dates must be in YYYY-MM-DD format
- Return ONLY valid JSON matching the schema — no prose, no markdown fences"""

_SCHEMA_HINT = """
JSON schema to follow exactly:
{
  "trip_name": "string",
  "destination": "string",
  "total_days": integer,
  "estimated_total_cost_usd": number,
  "generated_at": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "location": "string",
      "notes": "string or null",
      "flight": {"origin": "string", "destination": "string", "date": "YYYY-MM-DD", "duration_hours": number} or null,
      "hotel": {"city": "string", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "preferred_type": "string"} or null,
      "activities": [
        {"name": "string", "description": "string", "estimated_cost_usd": number or null,
         "duration_hours": number or null, "bookable": false}
      ]
    }
  ]
}
"""


def generate_itinerary_node(state: dict) -> dict:
    trip = TripRequest(**state["trip_request"])
    client = get_client()

    hp = trip.hotel_preferences
    fp = trip.flight_preferences

    user_msg = (
        f"Plan a {trip.num_nights}-night trip from {trip.origin} to {trip.destination}.\n"
        f"Dates: {trip.departure_date} to {trip.return_date}\n"
        f"Travelers: {trip.num_travelers}\n"
        f"Total budget: ${trip.budget_usd} ({trip.preferred_currency})\n"
        f"Style: {trip.trip_style or 'flexible'}\n"
        f"Accommodation: {trip.accommodation_type}\n"
        f"Hotel prefs: {hp.star_rating_min}–{hp.star_rating_max} stars, "
        f"max ${hp.max_price_per_night}/night, location: {hp.preferred_location or 'any'}\n"
        f"Flight prefs: max {fp.max_stops} stops, {fp.cabin_class} class\n"
        f"Special: {', '.join(trip.special_requirements) or 'none'}\n\n"
        f"{_SCHEMA_HINT}"
    )

    # If regenerating after rejection, include feedback
    messages = state.get("messages", [])
    if state.get("itinerary") and messages:
        recent_user = [m for m in messages[-4:] if m.get("role") == "user"]
        if recent_user:
            user_msg += f"\n\nUser feedback on previous plan: {recent_user[-1]['content']}"

    response_text = client.structured_output(
        messages=[{"role": "user", "content": user_msg}],
        system=_SYSTEM,
    )

    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        itinerary = Itinerary.model_validate_json(text)
    except Exception:
        data = json.loads(text)
        itinerary = Itinerary.model_validate(data)

    itinerary = itinerary.model_copy(update={"generated_at": datetime.now().strftime("%Y-%m-%d")})
    storage = create_storage()
    trip_id = state.get("trip_id")
    trip_data = {
        "profile_id": state.get("profile_id"),
        "trip_request": trip.model_dump(),
        "itinerary": itinerary.model_dump(),
        "status": "planned",
    }
    if trip_id:
        trip_data["trip_id"] = trip_id
    trip_id = storage.save_trip(trip_data)
    audit(
        storage,
        "trip.itinerary_generated",
        trip_id=trip_id,
        correlation_id=state.get("correlation_id"),
        data={"destination": trip.destination, "total_days": itinerary.total_days},
    )
    return {"itinerary": itinerary.model_dump(), "phase": "itinerary", "trip_id": trip_id}
