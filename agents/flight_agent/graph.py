"""Compiled LangGraph flight workflow and its public entry points.

    search_flights_for_trip(request)          start a run (new thread id)
    resume_flight_search(thread_id, answer)   answer the pending question, continue

Both return a FlightWorkflowOutcome: either a question to show the user
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
    FlightRecommendationResult,
    FlightSearchRequest,
    RecommendedFlight,
)
from .nodes import FlightWorkflowNodes
from .state import FlightState

logger = logging.getLogger(__name__)


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_clarify(state: FlightState) -> str:
    return "collect_preference_answer" if state.get("clarification_required") else "search_flights"


def route_after_search(state: FlightState) -> str:
    ok = not state.get("error") and bool(state.get("raw_offers"))
    return "rank_flights" if ok else "prepare_output"


def build_flight_graph(client=None, llm=None, checkpointer=None):
    """Compile the workflow. A checkpointer is required for the clarification interrupt."""
    n = FlightWorkflowNodes(client=client, llm=llm)
    g = StateGraph(FlightState)

    g.add_node("clarify_flight_preferences", n.clarify_flight_preferences_node)
    g.add_node("collect_preference_answer", n.collect_preference_answer_node)
    g.add_node("search_flights", n.search_flights_node)
    g.add_node("rank_flights", n.rank_flights_node)
    g.add_node("prepare_output", n.prepare_output_node)

    g.add_edge(START, "clarify_flight_preferences")
    g.add_conditional_edges(
        "clarify_flight_preferences", route_after_clarify,
        ["collect_preference_answer", "search_flights"],
    )
    g.add_edge("collect_preference_answer", "clarify_flight_preferences")
    g.add_conditional_edges(
        "search_flights", route_after_search,
        ["rank_flights", "prepare_output"],
    )
    g.add_edge("rank_flights", "prepare_output")
    g.add_edge("prepare_output", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


_default_graph = None
_lock = threading.Lock()


def get_flight_graph():
    """Process-wide graph, so paused threads survive across calls."""
    global _default_graph
    with _lock:
        if _default_graph is None:
            _default_graph = build_flight_graph()
        return _default_graph


# ── Public interface ─────────────────────────────────────────────────────────

@dataclass
class FlightWorkflowOutcome:
    thread_id: str
    status: str                                    # "needs_input" | "complete" | "error"
    question: str | None = None
    result: FlightRecommendationResult | None = None
    error: str | None = None

    @property
    def needs_input(self) -> bool:
        return self.status == "needs_input"


def _initial_state(request: FlightSearchRequest) -> FlightState:
    return {
        "origin": request.origin.strip().upper(),
        "destination": request.destination.strip().upper(),
        "departure_date": request.departure_date,
        "return_date": request.return_date,
        "num_passengers": max(request.num_passengers, 1),
        "budget_usd": request.budget_usd,
        "currency": request.currency or "USD",
        "preferences": request.preferences,
        "cabin_class": request.cabin_class,
        "max_stops": request.max_stops,
        "preferred_departure_period": request.preferred_departure_period,
        "clarification_attempts": 0,
        "clarification_history": [],
        "warnings": [],
        "error": None,
    }


def _outcome(thread_id: str, values: dict) -> FlightWorkflowOutcome:
    interrupts = values.get("__interrupt__")
    if interrupts:
        return FlightWorkflowOutcome(
            thread_id, "needs_input",
            question=interrupts[0].value["question"],
        )
    if values.get("error"):
        return FlightWorkflowOutcome(thread_id, "error", error=values["error"])

    result = FlightRecommendationResult(
        flights=[RecommendedFlight(**f) for f in (values.get("recommended_flights") or [])],
        candidates_found=values.get("candidates_found", 0),
        warnings=list(values.get("warnings") or []),
    )
    return FlightWorkflowOutcome(thread_id, "complete", result=result)


def search_flights_for_trip(
    request: FlightSearchRequest,
    *,
    thread_id: str | None = None,
    graph=None,
) -> FlightWorkflowOutcome:
    """Start the flight workflow; may stop early with a preference question."""
    graph = graph or get_flight_graph()
    thread_id = thread_id or uuid.uuid4().hex
    logger.info("Flight workflow start (thread %s): %s → %s", thread_id, request.origin, request.destination)
    values = graph.invoke(_initial_state(request), {"configurable": {"thread_id": thread_id}})
    return _outcome(thread_id, values)


def resume_flight_search(
    thread_id: str,
    answer: str,
    *,
    graph=None,
) -> FlightWorkflowOutcome:
    """Resume a paused workflow on the same thread with the user's answer."""
    graph = graph or get_flight_graph()
    logger.info("Flight workflow resume (thread %s): %r", thread_id, answer)
    values = graph.invoke(Command(resume=answer), {"configurable": {"thread_id": thread_id}})
    return _outcome(thread_id, values)
