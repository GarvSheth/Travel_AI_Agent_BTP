"""LLM generation of the next clarification question."""
from __future__ import annotations

import json

from schemas.trip_schema import TripExtraction
from workflows.input_pipeline.validator import ValidationResult

QUESTION_SCHEMA: dict = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

_SYSTEM = """You are a friendly travel-planning assistant collecting trip details.
Write exactly ONE short, natural question for the user.

Rules:
- Ask only about the fields listed under "fields_to_ask_about". Never ask about anything
  already present in current_trip_state.
- If invalid_fields is non-empty, resolve those first: briefly say what was wrong and ask
  for a corrected value.
- You may combine closely related fields into one question (e.g. departure date and
  return date) when it reads naturally, but keep it to a single question sentence.
- Use the conversation history so you don't repeat a question the user already answered.
- When "mode" is "optional_preferences", invite the user to share any of those
  preferences in one question and make clear they can skip.
- Return JSON: {"question": "..."}"""


def generate_question(
    *,
    original_prompt: str,
    trip: TripExtraction,
    validation: ValidationResult,
    history: list[dict],
) -> str:
    """Return one LLM-generated question. Raises on LLM failure or empty output."""
    from services.llm_client import get_client

    optional_mode = validation.is_valid
    context = {
        "mode": "optional_preferences" if optional_mode else "required_details",
        "original_prompt": original_prompt,
        "current_trip_state": {
            k: v for k, v in trip.model_dump().items() if v not in (None, [], {})
        },
        "fields_to_ask_about": (
            validation.unfilled_optional_fields if optional_mode
            else validation.invalid_fields + validation.missing_fields
        ),
        "missing_fields": validation.missing_fields,
        "invalid_fields": validation.invalid_fields,
        "validation_errors": validation.errors,
        "conversation_history": history,
    }
    raw = get_client().structured_output(
        messages=[{"role": "user", "content": json.dumps(context, default=str)}],
        system=_SYSTEM,
        schema=QUESTION_SCHEMA,
    )
    question = (json.loads(raw).get("question") or "").strip() if raw else ""
    if not question:
        raise ValueError("LLM returned an empty question")
    return question
