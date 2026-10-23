from models.booking import FlightOption, HotelOption
from models.trip import TripRequest
from services.amadeus_client import TravelSearchClient
from services.confidence_scorer import rank_options


def _get_search_client(trip: TripRequest) -> TravelSearchClient:
    from services.credential_manager import CredentialManager
    cm = CredentialManager()
    try:
        return TravelSearchClient(
            api_key=cm.get_credential("AMADEUS_API_KEY"),
            api_secret=cm.get_credential("AMADEUS_API_SECRET"),
        )
    except Exception:
        return TravelSearchClient()


def search_options_node(state: dict) -> dict:
    idx = state["current_booking_index"]
    item = state["bookable_items"][idx]
    trip = TripRequest(**state["trip_request"])
    client = _get_search_client(trip)

    raw: list[FlightOption | HotelOption]

    if item["type"] == "flight":
        data = item["data"]
        raw = client.search_flights(
            origin=trip.origin,
            destination=trip.destination,
            date=data.get("date", trip.departure_date),
            adults=trip.num_travelers,
        )
        prefs = trip.flight_preferences
    else:
        data = item["data"]
        city = data.get("city", trip.destination)[:3].upper()
        raw = client.search_hotels(
            city_code=city,
            check_in=data.get("check_in", trip.departure_date),
            check_out=data.get("check_out", trip.return_date),
            adults=trip.num_travelers,
        )
        prefs = trip.hotel_preferences

    scored = rank_options(raw, prefs)

    # Auto-relax if all options failed hard filters
    if not scored and item["type"] == "hotel" and prefs.max_price_per_night:
        relaxed_prefs = prefs.model_copy(
            update={"max_price_per_night": prefs.max_price_per_night * 1.2}
        )
        scored = rank_options(raw, relaxed_prefs)
        # Mark them so UI can show [RELAXED] badge
        for s in scored:
            if isinstance(s.option, HotelOption):
                s.option.source += " [relaxed]"

    new_results = dict(state["search_results"])
    new_results[str(idx)] = [s.model_dump() for s in scored]
    return {"search_results": new_results}
