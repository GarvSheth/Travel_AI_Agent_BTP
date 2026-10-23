from langgraph.types import interrupt

from models.trip import FlightPreferences, HotelPreferences, TripRequest
from services.llm_client import get_client
from services.persistence import create_storage
from services.profile_service import audit, profile_id, save_profile

_CORE_SYSTEM = """You are an expert travel planning assistant. Your job is to gather all the
details needed to plan the perfect trip.

Ask the user questions naturally — one or two at a time, not as a form. Be warm and helpful.
Once you have collected ALL required fields (destination, origin city, departure date, return date,
number of travelers, and total budget), call the `record_core_trip` tool immediately.

Required fields:
- destination (city or country)
- origin (departure city, IATA code if possible)
- departure_date (YYYY-MM-DD)
- return_date (YYYY-MM-DD)
- num_travelers (integer)
- budget_usd (total trip budget in USD)

Optional but helpful: preferred_currency, accommodation_type, trip_style, special_requirements."""

_PREF_SYSTEM = """You are an expert travel planning assistant helping a user refine their hotel
and flight preferences.

The user has already provided core trip details. Now ask follow-up questions to understand their
filtering preferences. Be conversational — infer sensible defaults from what they've said
(e.g., a budget of ₹4000/night implies 3–4 star hotels).

Ask about:
HOTELS: star rating range, max price per night, minimum guest rating (e.g. 4.5/5),
minimum number of reviews (e.g. 500+), preferred location (city center / near attractions /
near beach / airport), must-have amenities (wifi, pool, gym, breakfast, parking, ac).

FLIGHTS: direct or connecting, cabin class, preferred departure time window, max flight price,
any preferred airlines.

If the user says they have no strong preference for something, use sensible defaults.
Once you have enough information (even partial preferences are fine), call the
`record_preference_profile` tool."""

CORE_TOOL = {
    "name": "record_core_trip",
    "description": "Call once all required trip details are confirmed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "origin": {"type": "string", "description": "Departure city, IATA code preferred"},
            "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
            "return_date": {"type": "string", "description": "YYYY-MM-DD"},
            "num_travelers": {"type": "integer", "minimum": 1, "maximum": 20},
            "budget_usd": {"type": "number"},
            "preferred_currency": {"type": "string", "default": "USD"},
            "accommodation_type": {
                "type": "string",
                "enum": ["hotel", "hostel", "airbnb", "any"],
            },
            "trip_style": {"type": "string"},
            "special_requirements": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["destination", "origin", "departure_date", "return_date",
                     "num_travelers", "budget_usd"],
    },
}

PREF_TOOL = {
    "name": "record_preference_profile",
    "description": "Call after gathering hotel and flight preferences.",
    "input_schema": {
        "type": "object",
        "properties": {
            "hotel": {
                "type": "object",
                "properties": {
                    "star_rating_min": {"type": "integer"},
                    "star_rating_max": {"type": "integer"},
                    "max_price_per_night": {"type": "number"},
                    "min_rating": {"type": "number"},
                    "min_review_count": {"type": "integer"},
                    "preferred_location": {
                        "type": "string",
                        "enum": ["city_center", "near_attractions", "near_beach", "airport", "any"],
                    },
                    "required_amenities": {"type": "array", "items": {"type": "string"}},
                    "room_type": {"type": "string"},
                },
            },
            "flight": {
                "type": "object",
                "properties": {
                    "max_price": {"type": "number"},
                    "max_stops": {"type": "integer"},
                    "cabin_class": {
                        "type": "string",
                        "enum": ["economy", "business", "first"],
                    },
                    "preferred_airlines": {"type": "array", "items": {"type": "string"}},
                    "preferred_departure_time": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "any"],
                    },
                    "max_layover_hours": {"type": "number"},
                },
            },
        },
        "required": ["hotel", "flight"],
    },
}


def gather_preferences_node(state: dict) -> dict:
    phase = state.get("phase", "gathering_core")
    messages = list(state.get("messages", []))
    client = get_client()
    current_profile_id = state.get("profile_id") or profile_id()

    if phase == "gathering_core":
        if not messages:
            messages.append({"role": "user", "content": "I want to plan a trip."})

        resp = client.chat(messages, system=_CORE_SYSTEM, tools=[CORE_TOOL])
        tool_input = resp.get_tool_call("record_core_trip")

        if tool_input:
            trip = TripRequest(**tool_input)
            if resp.text:
                messages.append({"role": "assistant", "content": resp.text})
            return {
                "messages": messages,
                "trip_request": trip.model_dump(),
                "phase": "gathering_preferences",
                "profile_id": current_profile_id,
            }

        messages.append({"role": "assistant", "content": resp.text})
        user_reply = interrupt({"prompt": "You: ", "message": f"\n[bold cyan]Agent:[/bold cyan] {resp.text}"})
        messages.append({"role": "user", "content": user_reply})
        return {"messages": messages}

    else:  # gathering_preferences
        trip_dict = state.get("trip_request", {})
        context = (
            f"The user is travelling from {trip_dict.get('origin')} to "
            f"{trip_dict.get('destination')}, {trip_dict.get('departure_date')} – "
            f"{trip_dict.get('return_date')}, {trip_dict.get('num_travelers')} traveler(s), "
            f"budget ${trip_dict.get('budget_usd')}."
        )
        system = _PREF_SYSTEM + f"\n\nTrip context: {context}"

        if not messages or messages[-1]["role"] != "user":
            messages.append({
                "role": "user",
                "content": "Now let's set my hotel and flight preferences.",
            })

        resp = client.chat(messages, system=system, tools=[PREF_TOOL])
        tool_input = resp.get_tool_call("record_preference_profile")

        if tool_input:
            trip = TripRequest(**trip_dict)
            hotel_data = tool_input.get("hotel", {})
            flight_data = tool_input.get("flight", {})
            trip = trip.model_copy(update={
                "hotel_preferences": HotelPreferences(**hotel_data),
                "flight_preferences": FlightPreferences(**flight_data),
            })
            storage = create_storage()
            save_profile(
                {
                    "display_name": "Local traveler",
                    "preferences": {
                        "hotel_preferences": trip.hotel_preferences.model_dump(),
                        "flight_preferences": trip.flight_preferences.model_dump(),
                        "preferred_currency": trip.preferred_currency,
                    },
                },
                storage=storage,
            )
            audit(
                storage,
                "profile.preferences_updated",
                correlation_id=state.get("correlation_id"),
                data=trip.model_dump(),
            )
            if resp.text:
                messages.append({"role": "assistant", "content": resp.text})
            return {
                "messages": messages,
                "trip_request": trip.model_dump(),
                "phase": "itinerary",
                "profile_id": current_profile_id,
            }

        messages.append({"role": "assistant", "content": resp.text})
        user_reply = interrupt({"prompt": "You: ", "message": f"\n[bold cyan]Agent:[/bold cyan] {resp.text}"})
        messages.append({"role": "user", "content": user_reply})
        return {"messages": messages}
