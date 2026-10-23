from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.state import TravelAgentState
from agents.nodes.preference_node import gather_preferences_node
from agents.nodes.itinerary_node import generate_itinerary_node
from agents.nodes.present_itinerary_node import present_itinerary_node
from agents.nodes.await_approval_node import await_approval_node
from agents.nodes.extract_items_node import extract_items_node
from agents.nodes.check_items_node import check_items_node
from agents.nodes.search_node import search_options_node
from agents.nodes.present_options_node import present_options_node
from agents.nodes.await_decision_node import await_decision_node
from agents.nodes.booking_node import simulate_booking_node
from agents.nodes.increment_node import increment_index_node
from agents.nodes.summary_node import trip_summary_node


def build_graph(use_sqlite: bool = True):
    builder = StateGraph(TravelAgentState)

    builder.add_node("gather_preferences", gather_preferences_node)
    builder.add_node("generate_itinerary", generate_itinerary_node)
    builder.add_node("present_itinerary", present_itinerary_node)
    builder.add_node("await_approval", await_approval_node)
    builder.add_node("extract_items", extract_items_node)
    builder.add_node("check_items", check_items_node)
    builder.add_node("search_options", search_options_node)
    builder.add_node("present_options", present_options_node)
    builder.add_node("await_decision", await_decision_node)
    builder.add_node("simulate_booking", simulate_booking_node)
    builder.add_node("increment_index", increment_index_node)
    builder.add_node("trip_summary", trip_summary_node)

    # Entry
    builder.add_edge(START, "gather_preferences")

    # Preference loop: repeat until phase == "itinerary"
    builder.add_conditional_edges(
        "gather_preferences",
        lambda s: "generate_itinerary" if s.get("phase") == "itinerary" else "gather_preferences",
        {"generate_itinerary": "generate_itinerary", "gather_preferences": "gather_preferences"},
    )

    # Itinerary chain
    builder.add_edge("generate_itinerary", "present_itinerary")
    builder.add_edge("present_itinerary", "await_approval")

    # Approval: approved → booking; rejected → regenerate
    builder.add_conditional_edges(
        "await_approval",
        lambda s: "extract_items" if s.get("plan_approved") else "generate_itinerary",
        {"extract_items": "extract_items", "generate_itinerary": "generate_itinerary"},
    )

    # Booking loop entry
    builder.add_edge("extract_items", "check_items")

    builder.add_conditional_edges(
        "check_items",
        lambda s: (
            "trip_summary"
            if s["current_booking_index"] >= len(s["bookable_items"])
            else "search_options"
        ),
        {"trip_summary": "trip_summary", "search_options": "search_options"},
    )

    builder.add_edge("search_options", "present_options")
    builder.add_edge("present_options", "await_decision")

    builder.add_conditional_edges(
        "await_decision",
        lambda s: s.get("item_decision", "skip"),
        {
            "book": "simulate_booking",
            "skip": "increment_index",
            "show_more": "search_options",
        },
    )

    builder.add_edge("simulate_booking", "increment_index")
    builder.add_edge("increment_index", "check_items")
    builder.add_edge("trip_summary", END)

    # Checkpointer
    if use_sqlite:
        try:
            from langgraph_checkpoint_sqlite import SqliteSaver
            from config.settings import CHECKPOINT_DB
            CHECKPOINT_DB.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            checkpointer = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        except ImportError:
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
