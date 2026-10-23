from models.trip import Itinerary
from ui.console import print_itinerary


def present_itinerary_node(state: dict) -> dict:
    itinerary = Itinerary.model_validate(state["itinerary"])
    print_itinerary(itinerary)
    return {}
