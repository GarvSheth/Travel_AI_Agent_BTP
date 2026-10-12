"""LLM extraction of trip details from the initial prompt and every later answer."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Optional

from pydantic import ValidationError

from schemas.trip_schema import TripExtraction

SCALAR_FIELDS = [
    "destination", "origin", "departure_date", "return_date",
    "duration_min_days", "duration_max_days", "num_travelers",
    "budget", "currency", "trip_style", "accommodation_type",
]
LIST_FIELDS = ["interests", "traveler_details", "requested_tasks"]
TASKS = ["transport", "accommodation", "experiences", "itinerary"]
# Classified intent rather than a stated fact, so it can't have a verbatim quote;
# it is constrained by the TASKS enum instead.
UNQUOTED_FIELDS = {"requested_tasks"}

FIELDS = SCALAR_FIELDS + LIST_FIELDS
_INT_FIELDS = {"duration_min_days", "duration_max_days", "num_travelers"}
_FLOAT_FIELDS = {"budget"}
_PLACEHOLDERS = {"", "null", "none", "n/a", "unknown"}

# A flat list of facts, one per item, each carrying field + value + supporting quote
# together. A wide object of optional per-field properties made the model stop after
# a few fields; walking the message as a list extracts all of them reliably.
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": FIELDS},
                    "value": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["field", "value", "quote"],
            },
        },
    },
    "required": ["facts"],
}

_SYSTEM = """You extract structured travel-planning details from a conversation.
Today's date is {today}.

Original trip request: "{original_prompt}"
Trip details already collected: {current_state}
Conversation so far:
{history}
Question the user is answering: {question}

Read the user's LATEST message phrase by phrase and extract EVERY field it provides —
not only the field the question was about.

Return {{"facts": [{{"field": ..., "value": ..., "quote": ...}}, ...]}} with one fact per
field the message provides. quote = the exact words copied from the LATEST message that
state it; facts whose quote is not in the latest message are discarded.
A single message often contains many fields — go through it phrase by phrase and
return a fact for EVERY one (destination, origin, departure_date, return_date,
duration, num_travelers, budget, currency, preferences, ...).

Value formats:
- departure_date, return_date: YYYY-MM-DD
- duration_min_days, duration_max_days, num_travelers: integer digits only, e.g. "3"
- budget: number digits only, e.g. "20000"
- currency: ISO 4217 code, e.g. "INR"
- accommodation_type: one of hotel, hostel, airbnb, any
- interests, traveler_details, requested_tasks: one fact per item

RULES:
1. Never fabricate. Only return values the user actually stated or that follow directly
   from what they stated (e.g. "next Friday" -> a concrete date using today's date).
   Never infer a value from another field (no currency from a city, no style from a
   destination).
2. Omit fields the latest message does not mention. Never return null to clear a value.
3. If the user corrects a previously collected value, return the corrected value.
4. Dates are YYYY-MM-DD. If the user gives a date without a year, use the current year
   ({year}) — do not roll it forward; the system will ask the user if that date has passed.
5. Trip length goes in duration_min_days/duration_max_days ("3-4 days" -> 3 and 4;
   "5 days" -> 5 and 5). Do NOT turn a duration into departure/return dates, and when
   the user states both a duration and explicit dates, return both exactly as stated.
   Every "N days"/"N nights"/"a week" phrase in the latest message MUST be returned as
   duration_min_days and duration_max_days, even if a range is already known — e.g. "4 days"
   when 3-4 is known returns 4 and 4. Never put a duration in trip_style.
6. currency only when the user states it or uses an unambiguous symbol/word
   ($ = USD, ₹/rs/rupees = INR, € = EUR, £ = GBP). Never assume a currency.
   budget is the number alone ("20000 rs" -> budget 20000, currency INR).
7. "solo"/"just me" means num_travelers 1; "couple"/"me and my wife" means 2.
8. If the user declines a preference ("no preference", "skip", "none"), omit that field.
9. For list fields return only the items the latest message adds.
10. requested_tasks: only when the user explicitly limits what they want (e.g. "just book
    a hotel" -> ["accommodation"]). For a general trip-planning request, omit it.
11. Return raw JSON only."""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def call_llm_extraction(
    latest_message: str,
    *,
    original_prompt: str,
    current: TripExtraction,
    history: list[dict],
    question: Optional[str],
    today: Optional[date] = None,
) -> dict:
    """Ask the LLM for every field supported by the latest message. Raises on LLM failure."""
    from services.llm_client import get_client

    known = {k: v for k, v in current.model_dump().items() if v not in (None, [], {})}
    today = today or date.today()
    system = _SYSTEM.format(
        today=today.isoformat(),
        year=today.year,
        original_prompt=original_prompt,
        current_state=json.dumps(known, default=str),
        history=_format_history(history),
        question=question or "(none — this is the initial request)",
    )
    raw = get_client().structured_output(
        messages=[{"role": "user", "content": latest_message}],
        system=system,
        schema=EXTRACTION_SCHEMA,
    )
    data = json.loads(raw) if raw else {}
    if not isinstance(data, dict):
        raise ValueError(f"Extraction must be a JSON object, got {type(data).__name__}")
    return parse_extraction(data, latest_message)


def _squash(text: str) -> str:
    return " ".join(str(text).lower().split())


def _coerce(field: str, value: str):
    """Best-effort typing; on failure the raw string is kept so the schema rejects it
    and the validator reports it as invalid instead of it vanishing silently."""
    try:
        if field in _INT_FIELDS:
            return int(float(value))
        if field in _FLOAT_FIELDS:
            return float(value)
    except ValueError:
        pass
    return value


def parse_extraction(data: dict, latest_message: str) -> dict:
    """Turn {"facts": [{field, value, quote}]} into typed updates.

    A fact is kept when its value is non-empty and its quote occurs verbatim in the
    latest message (requested_tasks is instead restricted to the TASKS enum).
    Every discarded fact is logged with the reason.
    """
    message = _squash(latest_message)
    updates: dict = {}
    dropped: list[str] = []
    for fact in data.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        field = fact.get("field")
        value = str(fact.get("value") or "").strip()
        quote = str(fact.get("quote") or "")
        if field not in FIELDS:
            dropped.append(f"{field}: unknown field")
            continue
        if value.lower() in _PLACEHOLDERS:
            continue
        if field in UNQUOTED_FIELDS:
            if value not in TASKS:
                dropped.append(f"{field}: {value!r} is not a known task")
                continue
        elif not quote or _squash(quote) not in message:
            dropped.append(f"{field}={value!r}: quote {quote!r} not found in message")
            continue
        if field in LIST_FIELDS:
            items = updates.setdefault(field, [])
            if value not in items:
                items.append(value)
        else:
            updates[field] = _coerce(field, value)
    if dropped:
        print(f"[extraction] dropped facts: {dropped}", flush=True)
    return updates


def _normalize(field: str, value):
    if isinstance(value, str):
        value = value.strip()
    if field == "currency" and isinstance(value, str):
        return value.upper()
    if field == "accommodation_type" and isinstance(value, str):
        return value.lower()
    return value


def merge_updates(current: TripExtraction, updates: dict) -> tuple[TripExtraction, list[dict]]:
    """Apply LLM updates field-by-field.

    Scalars overwrite (handles corrections); lists are unioned. A value that fails
    schema validation is not applied and is reported in the rejected list instead.
    """
    data = current.model_dump()
    rejected: list[dict] = []
    applied: set[str] = set()

    for field, value in updates.items():
        value = _normalize(field, value)
        if field in LIST_FIELDS:
            items = value if isinstance(value, list) else [value]
            merged = list(data[field])
            merged += [i for i in items if i not in merged]
            candidate = {**data, field: merged}
        else:
            candidate = {**data, field: value}
        try:
            data = TripExtraction.model_validate(candidate).model_dump()
            applied.add(field)
        except ValidationError as exc:
            rejected.append({
                "field": field,
                "value": value,
                "error": exc.errors()[0].get("msg", str(exc)),
            })

    _reconcile_dates(data, applied)
    return TripExtraction.model_validate(data), rejected


def _parse(d: Optional[str]) -> Optional[date]:
    return datetime.strptime(d, "%Y-%m-%d").date() if d else None


def _reconcile_dates(data: dict, applied: set[str]) -> None:
    """Derive return_date or duration when the other side is fully determined.

    A trip of N days departing on D returns on D + (N - 1).
    """
    dep, ret = _parse(data["departure_date"]), _parse(data["return_date"])
    dmin, dmax = data["duration_min_days"], data["duration_max_days"]
    if dmin is not None and dmax is None:
        data["duration_max_days"] = dmax = dmin
    fixed_duration = dmin if dmin is not None and dmin == dmax else None
    duration_changed = bool({"duration_min_days", "duration_max_days"} & applied)

    if dep and ret and "return_date" in applied and not duration_changed and ret >= dep:
        span = (ret - dep).days + 1
        data["duration_min_days"] = data["duration_max_days"] = span
    elif (dep and fixed_duration and "return_date" not in applied
          and (ret is None or "departure_date" in applied or duration_changed)):
        data["return_date"] = (dep + timedelta(days=fixed_duration - 1)).isoformat()
