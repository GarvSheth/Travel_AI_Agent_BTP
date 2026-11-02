"""LLM step for flight preference clarification.

One LLM call per turn judges whether the traveller's cabin class, stop
preference, and departure timing are clear. If any are missing it writes one
short, friendly question that asks for all of them at once. A second call
parses the answer back into structured values.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CLARIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["clear", "unclear"]},
        "cabin_class": {
            "type": "string",
            "enum": ["economy", "premium_economy", "business", "first", "any", ""],
        },
        "max_stops": {"type": "integer"},
        "preferred_departure_period": {
            "type": "string",
            "enum": ["morning", "afternoon", "evening", "any", ""],
        },
        "question": {"type": "string"},
    },
    "required": ["status", "cabin_class", "max_stops", "preferred_departure_period", "question"],
}

_SYSTEM = """You help a traveller set their flight preferences before searching.

You receive the origin airport, destination airport, the traveller's stated
preferences (may be empty), and any prior answers they gave (may be empty).

Your job:
1. Determine whether all three preferences are known:
   - cabin_class: "economy", "premium_economy", "business", or "first"
   - max_stops: 0 (direct only), 1 (at most one stop), or 2 (any number)
   - preferred_departure_period: "morning" (before noon), "afternoon" (noon–5 pm),
     "evening" (after 5 pm), or "any"

2. If the traveller's text already states or clearly implies all three, set
   status = "clear" and fill the three fields. Use "any" / 2 for open preferences.

3. If any preference is still unknown after reading the conversation, set
   status = "unclear" and write ONE short, friendly question that asks for all
   missing preferences at once. Offer concrete options for each (e.g. "economy,
   business, or first class?"). question = "" when status = "clear".

4. On the second or later turn, always set status = "clear" — do not keep asking
   after the traveller has answered once, even if the answer is vague. Treat
   vague answers as "any" / "economy" / 2.

Rules:
- Never invent a preference the traveller did not express.
- Never set status = "unclear" more than once per conversation.
- max_stops must be an integer: 0, 1, or 2.
- If the traveller says "no preference" or "flexible" for a field, use "any" / 2.

Return JSON:
{"status", "cabin_class", "max_stops", "preferred_departure_period", "question"}"""


@dataclass
class FlightPreferenceDecision:
    status: str                              # "clear" | "unclear"
    cabin_class: Optional[str] = None        # "economy"|"premium_economy"|"business"|"first"|"any"
    max_stops: Optional[int] = None          # 0|1|2
    preferred_departure_period: Optional[str] = None  # "morning"|"afternoon"|"evening"|"any"
    question: str = ""


def decide_flight_preferences(
    *,
    origin: str,
    destination: str,
    preferences: str,
    answers: list[dict],
    llm=None,
) -> FlightPreferenceDecision:
    """Ask the LLM whether flight preferences are clear. Raises ValueError on bad output."""
    if llm is None:
        from services.llm_client import get_client
        llm = get_client()

    context = {
        "origin": origin,
        "destination": destination,
        "traveller_preferences": preferences or None,
        "preference_conversation": answers,
    }
    raw = llm.structured_output(
        messages=[{"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        system=_SYSTEM,
        schema=CLARIFY_SCHEMA,
    )

    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM returned invalid JSON for flight preferences.") from exc

    status = str(data.get("status") or "").strip()
    if status not in {"clear", "unclear"}:
        raise ValueError(f"LLM returned unknown preference status {status!r}.")

    question = str(data.get("question") or "").strip()
    if status == "unclear" and not question:
        raise ValueError("LLM returned status=unclear but wrote no question.")

    cabin = str(data.get("cabin_class") or "").strip() or None
    if cabin in ("", "any"):
        cabin = "economy"

    raw_stops = data.get("max_stops")
    try:
        stops = int(raw_stops) if raw_stops is not None else 2
        stops = max(0, min(stops, 2))
    except (TypeError, ValueError):
        stops = 2

    period = str(data.get("preferred_departure_period") or "").strip() or "any"
    if period not in {"morning", "afternoon", "evening", "any"}:
        period = "any"

    logger.info(
        "Flight preference decision %s → %s: cabin=%s stops=%s period=%s",
        origin, destination, cabin, stops, period,
    )
    return FlightPreferenceDecision(
        status=status,
        cabin_class=cabin if status == "clear" else None,
        max_stops=stops if status == "clear" else None,
        preferred_departure_period=period if status == "clear" else None,
        question=question,
    )
