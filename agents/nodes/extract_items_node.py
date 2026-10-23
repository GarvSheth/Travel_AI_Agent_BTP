from models.trip import Itinerary


def extract_items_node(state: dict) -> dict:
    itinerary = Itinerary.model_validate(state["itinerary"])
    items = []
    for i, day in enumerate(itinerary.days):
        if day.flight:
            items.append({"type": "flight", "day_index": i, "data": day.flight.model_dump()})
        if day.hotel:
            items.append({"type": "hotel", "day_index": i, "data": day.hotel.model_dump()})

    return {
        "bookable_items": items,
        "current_booking_index": 0,
        "search_results": {},
        "booking_records": [],
        "phase": "booking",
    }
