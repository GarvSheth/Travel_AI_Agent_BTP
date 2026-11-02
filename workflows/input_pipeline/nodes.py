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


def extract_node(state: InputPipelineState) -> dict:
    history = state.get("conversation_history", [])
    message = state["latest_user_message"]
    current = _trip(state)

    try:
        updates = call_llm_extraction(
            message,
            original_prompt=state["original_prompt"],
            current=current,
            history=history[:-1],  # history already ends with `message`
            question=state.get("pending_question"),
        )
    except Exception:
        updates = {}

    updated, rejected = merge_updates(current, updates)

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
