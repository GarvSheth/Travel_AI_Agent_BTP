from models.trip import TripRequest
from ui.console import console, print_flight_options, print_hotel_options


def present_options_node(state: dict) -> dict:
    idx = state["current_booking_index"]
    item = state["bookable_items"][idx]
    scored = state["search_results"].get(str(idx), [])
    trip = TripRequest(**state["trip_request"])

    total = len(state["bookable_items"])
    console.print(f"\n[bold]Booking item {idx + 1} of {total}:[/bold] {item['type'].title()}")

    if not scored:
        console.print("[yellow]  No options found matching your filters. Skipping.[/yellow]")
        return {}

    if item["type"] == "flight":
        print_flight_options(scored, idx, trip)
    else:
        print_hotel_options(scored, idx, trip)

    return {}
