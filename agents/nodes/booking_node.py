from models.trip import TripRequest
from services.booking_service import BookingUnavailableError, create_booking_provider
from services.persistence import create_storage
from services.profile_service import audit


def simulate_booking_node(state: dict) -> dict:
    """Start a real provider checkout; never claim a booking from local simulation."""
    idx = state["current_booking_index"]
    selected = state["search_results"].get(f"{idx}_selected", {})
    item = state["bookable_items"][idx]
    trip = TripRequest(**state["trip_request"])
    try:
        checkout_url = create_booking_provider().create_checkout(
            selected,
            idempotency_key=f"{state.get('trip_id', 'trip')}-{idx}",
        )
    except BookingUnavailableError as exc:
        storage = create_storage()
        audit(
            storage,
            "booking.checkout_unavailable",
            trip_id=state.get("trip_id"),
            correlation_id=state.get("correlation_id"),
            data={"item_type": item["type"], "error": str(exc)},
        )
        return {"error_message": str(exc), "item_decision": "skip"}

    storage = create_storage()
    audit(
        storage,
        "booking.checkout_created",
        trip_id=state.get("trip_id"),
        correlation_id=state.get("correlation_id"),
        data={"item_type": item["type"], "checkout_url": checkout_url},
    )
    return {"error_message": f"Continue checkout at: {checkout_url}"}
