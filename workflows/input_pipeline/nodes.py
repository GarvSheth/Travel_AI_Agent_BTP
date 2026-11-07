"""LangGraph nodes for the input-collection workflow."""
from __future__ import annotations

from langgraph.types import interrupt

from schemas.trip_schema import TripExtraction
from workflows.input_pipeline.extraction import call_llm_extraction, merge_updates
from workflows.input_pipeline.payload import build_orchestrator_input
from workflows.input_pipeline.question_generator import generate_question
from workflows.input_pipeline.state import InputPipelineState
from workflows.input_pipeline.validator import validate_trip


def _trip(state: InputPipelineState) -> TripExtraction:
    return TripExtraction.model_validate(state.get("trip_extraction") or {})


def _print_extraction(message: str, updates: dict, rejected: list[dict],
                      trip: TripExtraction, error: Exception | None) -> None:
    """Show in the terminal what was fetched from the user's latest message."""
    lines = ["", "─" * 70, f"[input_pipeline] user message: {message!r}"]
    if error is not None:
        lines.append(f"  extraction FAILED: {type(error).__name__}: {error}")
    elif updates:
        lines.append("  fetched:")
        lines += [f"    {field:20} = {value!r}" for field, value in updates.items()]
    else:
        lines.append("  fetched: (nothing)")
    for r in rejected:
        lines.append(f"  rejected: {r['field']} = {r['value']!r} ({r['error']})")
    known = {k: v for k, v in trip.model_dump().items() if v not in (None, [], {})}
    lines.append("  trip so far:")
    lines += [f"    {field:20} = {value!r}" for field, value in known.items()]
    lines.append("─" * 70)
    print("\n".join(lines), flush=True)


def extract_node(state: InputPipelineState) -> dict:
    history = state.get("conversation_history", [])
    message = state["latest_user_message"]
    current = _trip(state)

    error = None
    try:
        updates = call_llm_extraction(
            message,
            original_prompt=state["original_prompt"],
            current=current,
            history=history[:-1],  # history already ends with `message`
            question=state.get("pending_question"),
        )
    except Exception as exc:
        updates = {}
        error = exc

    updated, rejected = merge_updates(current, updates)
    _print_extraction(message, updates, rejected, updated, error)

    return {
        "trip_extraction": updated.model_dump(),
        "rejected_updates": rejected,
        "pending_question": None,
    }


def validate_node(state: InputPipelineState) -> dict:
    result = validate_trip(_trip(state), state.get("rejected_updates"))
    return {
        "missing_fields": result.missing_fields,
        "invalid_fields": result.invalid_fields,
        "validation_errors": result.errors,
    }


def ask_question_node(state: InputPipelineState) -> dict:
    trip = _trip(state)
    result = validate_trip(trip, state.get("rejected_updates"))
    history = state.get("conversation_history", [])
    try:
        question = generate_question(
            original_prompt=state["original_prompt"],
            trip=trip,
            validation=result,
            history=history,
        )
    except Exception:
        fields = result.invalid_fields + result.missing_fields or result.unfilled_optional_fields
        question = ("Could you tell me more about: "
                    + ", ".join(f.replace("_", " ") for f in fields) + "?")

    return {
        "pending_question": question,
        "optional_prompted": state.get("optional_prompted", False) or result.is_valid,
    }


def await_answer_node(state: InputPipelineState) -> dict:
    question = state["pending_question"]
    answer = interrupt({"question": question})
    history = list(state.get("conversation_history", []))
    history += [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]
    return {"latest_user_message": answer, "conversation_history": history}


def finalize_node(state: InputPipelineState) -> dict:
    oi = build_orchestrator_input(state["original_prompt"], _trip(state))
    payload = oi.model_dump()
    return {"final_payload": payload, "status": "complete", "pending_question": None}
