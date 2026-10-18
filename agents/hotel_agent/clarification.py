"""LLM steps for the hotel clarifications: where to stay, and which star category.

One LLM call per location turn judges whether the accommodation location is
clear and writes the follow-up question; once the location is settled the same
call also writes the star-category question (or reports the stars the traveller
already stated). A second, small call turns the star answer into star values.
The LLM never produces coordinates: only a place name the geocoder resolves.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CLARIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["specific", "city_centre", "unclear"]},
        "location_query": {"type": "string"},
        "question": {"type": "string"},
        "stars_mentioned": {"type": "array", "items": {"type": "number"}},
        "star_question": {"type": "string"},
    },
    "required": ["status", "location_query", "question", "stars_mentioned", "star_question"],
}

STAR_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "stars": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["stars"],
}

VALID_STARS = {x / 2 for x in range(2, 11)}          # 1.0, 1.5, ... 5.0

_SYSTEM = """You help a traveller choose WHERE in their destination city to stay.

You receive the destination, the traveller's stated preferences, and (if any) their
answers to earlier location questions.

Decide one status:
- "specific": the traveller named a landmark, neighbourhood, area, venue, airport, station
  or similar place inside the destination. Put a short, geocodable place name in
  location_query (e.g. "IIM Ahmedabad", "Juhu Beach", "Jaipur International Airport").
  Resolve relative phrases using the destination ("the airport" -> "<city> airport").
- "city_centre": the traveller wants the city centre / downtown, or says they have no
  preference / anywhere is fine. location_query = "".
- "unclear": no usable location preference has been given yet, or the latest answer is
  too vague to look up. location_query = "".

If status is "unclear", write ONE short, friendly question asking where they would like
to stay. Tailor it to the destination and to what they said (purpose of trip, interests,
earlier answers); you may suggest 2-3 plausible areas or landmarks in that city, plus
"the city centre, or somewhere else". Do not repeat an earlier question word for word.
Otherwise question = "".

Star category (hotel class):
- stars_mentioned: star categories the traveller has ALREADY stated anywhere in the input
  (e.g. "5-star" -> [5]; "4 or 5 star" -> [4, 5]; "at least 4 star" -> [4, 4.5, 5]).
  Empty list if they have not stated one. Never infer stars from budget or vibe.
- star_question: only when status is "specific" or "city_centre" AND stars_mentioned is
  empty, write ONE short, friendly question asking which hotel star category they prefer
  (e.g. 3-star, 4-star, 5-star) and that "no preference" is fine. Tailor it to the
  destination and what they said. Otherwise star_question = "".

Never invent a preference the traveller did not express.
Return JSON: {"status", "location_query", "question", "stars_mentioned", "star_question"}"""

_STAR_SYSTEM = """Convert a traveller's answer about hotel star category into star values.

Allowed values: 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.
- "4 star" -> [4]; "4 or 5" -> [4, 5]; "at least 4" / "4+" -> [4, 4.5, 5];
  "3 to 4" -> [3, 3.5, 4]; "luxury" / "five star" -> [5].
- No preference, "any", "don't know", "doesn't matter" or an unrelated answer -> [].
Return JSON: {"stars": [...]}"""


@dataclass
class LocationDecision:
    status: str                 # "specific" | "city_centre" | "unclear"
    location_query: str
    question: str
    stars_mentioned: list[float] = field(default_factory=list)
    star_question: str = ""


def clean_stars(values) -> list[float]:
    """Keep only valid star categories, sorted and de-duplicated."""
    out: set[float] = set()
    for v in values if isinstance(values, list) else []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f in VALID_STARS:
            out.add(f)
    return sorted(out)


def decide_accommodation_location(
    *,
    city: str,
    preferences: str,
    vibe: str,
    answers: list[dict],
    llm=None,
) -> LocationDecision:
    """Ask the LLM whether the stay location is clear. Raises ValueError on bad output."""

    if llm is None:
        from services.llm_client import get_client
        llm = get_client()

    context = {
        "destination": city,
        "traveller_preferences": preferences or None,
        "preferred_vibe": vibe or None,
        "location_conversation": answers,
    }
    raw = llm.structured_output(
        messages=[{"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        system=_SYSTEM,
        schema=CLARIFY_SCHEMA,
    )
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM returned invalid JSON for the location question.") from exc

    status = str(data.get("status") or "").strip()
    query = str(data.get("location_query") or "").strip()
    question = str(data.get("question") or "").strip()

    if status == "specific" and not query:
        status = "unclear"
    if status not in {"specific", "city_centre", "unclear"}:
        raise ValueError(f"LLM returned an unknown location status {status!r}.")
    if status == "unclear" and not question:
        raise ValueError("LLM asked for clarification but wrote no question.")

    stars = clean_stars(data.get("stars_mentioned"))
    star_question = str(data.get("star_question") or "").strip()

    logger.info("Location decision for %s: %s %r", city, status, query or question)
    return LocationDecision(status, query, question, stars, star_question)


def parse_star_answer(answer: str, llm=None) -> list[float]:
    """Star values from the traveller's answer ([] = no preference). Raises ValueError."""
    if not (answer or "").strip():
        return []
    if llm is None:
        from services.llm_client import get_client
        llm = get_client()
    raw = llm.structured_output(
        messages=[{"role": "user", "content": answer}],
        system=_STAR_SYSTEM,
        schema=STAR_SCHEMA,
    )
    try:
        return clean_stars(json.loads(raw).get("stars"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("LLM returned invalid JSON for the star rating.") from exc
