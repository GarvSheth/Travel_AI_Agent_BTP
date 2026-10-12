"""LangGraph workflow: extract -> validate -> (ask -> await answer -> extract ...) | finalize."""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from schemas.trip_schema import OrchestratorInput, TripExtraction
from workflows.input_pipeline.nodes import (
    ask_question_node,
    await_answer_node,
    extract_node,
    finalize_node,
    log_section,
    validate_node,
)
from workflows.input_pipeline.state import InputPipelineState
from workflows.input_pipeline.validator import validate_trip


def route_after_validation(state: InputPipelineState) -> str:
    """Only the deterministic validator decides whether input is complete."""
    if state.get("missing_fields") or state.get("invalid_fields"):
        return "ask_question"
    if not state.get("optional_prompted"):
        trip = TripExtraction.model_validate(state.get("trip_extraction") or {})
        if validate_trip(trip).unfilled_optional_fields:
            return "ask_question"
    return "finalize"


def build_input_graph(checkpointer=None):
    builder = StateGraph(InputPipelineState)
    builder.add_node("extract", extract_node)
    builder.add_node("validate", validate_node)
    builder.add_node("ask_question", ask_question_node)
    builder.add_node("await_answer", await_answer_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "validate")
    builder.add_conditional_edges(
        "validate", route_after_validation,
        {"ask_question": "ask_question", "finalize": "finalize"},
    )
    builder.add_edge("ask_question", "await_answer")
    builder.add_edge("await_answer", "extract")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


class TurnResult:
    """Outcome of one user turn: either a question to show or the final payload."""

    def __init__(self, question: Optional[str], payload: Optional[OrchestratorInput]):
        self.question = question
        self.payload = payload

    @property
    def complete(self) -> bool:
        return self.payload is not None


def _to_turn_result(result: dict) -> TurnResult:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return TurnResult(question=interrupts[0].value["question"], payload=None)
    return TurnResult(
        question=None,
        payload=OrchestratorInput.model_validate(result["final_payload"]),
    )


def start_conversation(graph, thread_id: str, prompt: str) -> TurnResult:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Travel prompt cannot be empty.")
    log_section("INITIAL USER INPUT", prompt)
    initial: InputPipelineState = {
        "original_prompt": prompt,
        "latest_user_message": prompt,
        "trip_extraction": {},
        "conversation_history": [{"role": "user", "content": prompt}],
        "pending_question": None,
        "optional_prompted": False,
        "status": "collecting_input",
        "final_payload": None,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return _to_turn_result(graph.invoke(initial, config))


def submit_answer(graph, thread_id: str, answer: str) -> TurnResult:
    config = {"configurable": {"thread_id": thread_id}}
    return _to_turn_result(graph.invoke(Command(resume=answer), config))


def _consumed_user_turns(graph, thread_id: str) -> int:
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    history = (snap.values or {}).get("conversation_history", [])
    return sum(1 for m in history if m["role"] == "user")


def current_turn(graph, thread_id: str) -> TurnResult:
    """Result for the thread's latest checkpoint, finishing an interrupted run if needed."""
    config = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(config)
    if snap.values.get("final_payload"):
        return _to_turn_result(snap.values)
    interrupts = [i for task in snap.tasks for i in task.interrupts]
    if interrupts:
        return TurnResult(question=interrupts[0].value["question"], payload=None)
    return _to_turn_result(graph.invoke(None, config))


def handle_user_message(graph, thread_id: str, text: str, turn_index: int) -> TurnResult:
    """Process the user's `turn_index`-th message (0 = initial prompt) exactly once.

    If the graph already consumed this turn (e.g. a Streamlit rerun replays it),
    return the current result instead of invoking the graph again.
    """
    if _consumed_user_turns(graph, thread_id) > turn_index:
        return current_turn(graph, thread_id)
    if turn_index == 0:
        return start_conversation(graph, thread_id, text)
    return submit_answer(graph, thread_id, text)
