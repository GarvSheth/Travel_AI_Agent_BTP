"""Compiled LangGraph hotel workflow and its public entry points.

    search_hotels(request)              start a run (new thread id)
    resume_hotel_search(thread_id, a)   answer the pending question, continue

Both return a HotelWorkflowOutcome: either a question to show the user
(status "needs_input") or the finished result / error. Nothing blocks waiting
for the user: the graph pauses at an interrupt, its state lives in the
checkpointer under the thread id, and a later call resumes it.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .agent import (
    HotelRecommendationResult,
    HotelSearchRequest,
    RecommendedHotel,
    ResolvedLocation,
)
from .nodes import HotelWorkflowNodes
from .state import HotelState

logger = logging.getLogger(__name__)


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_clarify(state: HotelState) -> str:
    return "collect_location_answer" if state.get("clarification_required") else "clarify_star_rating"


def route_after_resolve(state: HotelState) -> str:
    ok = not state.get("error") and state.get("latitude") is not None
    return "search_hotels" if ok else "prepare_output"


def route_after_search(state: HotelState) -> str:
    return "load_facilities" if state.get("hotels") and not state.get("error") else "prepare_output"


def route_after_rank(state: HotelState) -> str:
    return "get_hotel_details" if state.get("ranked_hotels") and not state.get("error") else "prepare_output"


def build_hotel_graph(client=None, llm=None, checkpointer=None):
    """Compile the workflow. A checkpointer is required for the clarification interrupt."""
    n = HotelWorkflowNodes(client=client, llm=llm)
    g = StateGraph(HotelState)
    g.add_node("clarify_accommodation_location", n.clarify_accommodation_location_node)
    g.add_node("collect_location_answer", n.collect_location_answer_node)
    g.add_node("clarify_star_rating", n.clarify_star_rating_node)
    g.add_node("resolve_location", n.resolve_location_node)
    g.add_node("search_hotels", n.search_hotels_node)
    g.add_node("load_facilities", n.load_facilities_node)
    g.add_node("rank_hotels", n.rank_hotels_node)
    g.add_node("get_hotel_details", n.get_hotel_details_node)
    g.add_node("prepare_output", n.prepare_output_node)

    g.add_edge(START, "clarify_accommodation_location")
    g.add_conditional_edges("clarify_accommodation_location", route_after_clarify,
                            ["collect_location_answer", "clarify_star_rating"])
    g.add_edge("collect_location_answer", "clarify_accommodation_location")
    g.add_edge("clarify_star_rating", "resolve_location")
    g.add_conditional_edges("resolve_location", route_after_resolve,
                            ["search_hotels", "prepare_output"])
    g.add_conditional_edges("search_hotels", route_after_search,
                            ["load_facilities", "prepare_output"])
    g.add_edge("load_facilities", "rank_hotels")
    g.add_conditional_edges("rank_hotels", route_after_rank,
                            ["get_hotel_details", "prepare_output"])
    g.add_edge("get_hotel_details", "prepare_output")
    g.add_edge("prepare_output", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


_default_graph = None
_lock = threading.Lock()


def get_hotel_graph():
    """Process-wide graph, so paused threads survive Streamlit reruns."""
    global _default_graph
    with _lock:
        if _default_graph is None:
            _default_graph = build_hotel_graph()
        return _default_graph


# ── Public interface ─────────────────────────────────────────────────────────

@dataclass
class HotelWorkflowOutcome:
    thread_id: str
    status: str                                   # "needs_input" | "complete" | "error"
    question: str | None = None
    result: HotelRecommendationResult | None = None
    error: str | None = None

    @property
    def needs_input(self) -> bool:
        return self.status == "needs_input"


def _initial_state(request: HotelSearchRequest) -> HotelState:
    return {
        "city": request.city.strip(),
        "country_code": (request.country_code or "").strip().upper(),
        "location": (request.location or "").strip() or None,
        "preferences": request.preferences,
        "mandatory_requirements": list(request.mandatory_requirements),
        "vibe": request.vibe,
        "star_rating": request.star_rating or None,   # None -> graph asks
        "min_rating": request.min_rating,
        "radius": request.radius,
        "clarification_attempts": 0,
        "clarification_history": [],
        "warnings": [],
        "error": None,
    }


def _outcome(thread_id: str, values: dict) -> HotelWorkflowOutcome:
    interrupts = values.get("__interrupt__")
    if interrupts:
        return HotelWorkflowOutcome(thread_id, "needs_input",
                                    question=interrupts[0].value["question"])
    if values.get("error"):
        return HotelWorkflowOutcome(thread_id, "error", error=values["error"])

    location = None
    if values.get("latitude") is not None:
        location = ResolvedLocation(values["latitude"], values["longitude"],
                                    values.get("location_label") or "",
                                    values.get("location_source") or "")
    result = HotelRecommendationResult(
        hotels=[RecommendedHotel(**h) for h in values.get("recommended_hotels") or []],
        candidates_found=values.get("candidates_found", 0),
        location=location,
        warnings=list(values.get("warnings") or []),
    )
    return HotelWorkflowOutcome(thread_id, "complete", result=result)


def search_hotels(request: HotelSearchRequest, *, thread_id: str | None = None,
                  graph=None) -> HotelWorkflowOutcome:
    """Start the hotel workflow; may stop early with a location question."""
    graph = graph or get_hotel_graph()
    thread_id = thread_id or uuid.uuid4().hex
    logger.info("Hotel workflow start (thread %s): %s", thread_id, request)
    values = graph.invoke(_initial_state(request), {"configurable": {"thread_id": thread_id}})
    return _outcome(thread_id, values)


def resume_hotel_search(thread_id: str, answer: str, *, graph=None) -> HotelWorkflowOutcome:
    """Resume a paused workflow on the same thread with the user's answer."""
    graph = graph or get_hotel_graph()
    logger.info("Hotel workflow resume (thread %s): %r", thread_id, answer)
    values = graph.invoke(Command(resume=answer), {"configurable": {"thread_id": thread_id}})
    return _outcome(thread_id, values)
