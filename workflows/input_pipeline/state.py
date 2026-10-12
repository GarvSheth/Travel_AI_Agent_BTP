"""Canonical state for the input-collection workflow."""
from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class InputPipelineState(TypedDict, total=False):
    original_prompt: str
    latest_user_message: str
    trip_extraction: dict                 # serialized TripExtraction — the only copy of trip data
    conversation_history: list[dict]      # [{"role": "assistant"|"user", "content": str}]
    pending_question: Optional[str]
    missing_fields: list[str]
    invalid_fields: list[str]
    validation_errors: list[str]
    rejected_updates: list[dict]          # LLM values that failed schema checks this turn
    optional_prompted: bool               # optional preferences are offered at most once
    status: str                           # "collecting_input" | "complete"
    final_payload: Optional[dict]         # OrchestratorInput.model_dump() once complete
