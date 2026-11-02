"""Flight agent data types.

The recommendation workflow itself (clarify preferences → search Duffel →
LLM rank → output) is a LangGraph graph: see ``graph.py`` / ``nodes.py``
and the public ``search_flights_for_trip(request)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlightSearchRequest:
    origin: str                              # IATA code, e.g. "BOM"
    destination: str                         # IATA code, e.g. "LHR"
    departure_date: str                      # "YYYY-MM-DD"
    return_date: Optional[str] = None        # "YYYY-MM-DD" or None for one-way
    num_passengers: int = 1
    budget_usd: Optional[float] = None
    currency: str = "USD"
    cabin_class: Optional[str] = None        # None → graph asks the user
    max_stops: Optional[int] = None          # None → graph asks; 0=direct, 1, 2=any
    preferred_departure_period: Optional[str] = None  # "morning"|"afternoon"|"evening"|"any"
    preferences: str = ""                    # the traveller's own words


@dataclass
class RecommendedFlight:
    flight_id: str
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    duration: str
    price_usd: float
    currency: str
    stops: int
    departure_period: str
    cabin_class: str
    rank: int
    match_score: int          # 0-100
    explanation: str
    source: str = "duffel"


@dataclass
class FlightRecommendationResult:
    flights: list[RecommendedFlight]
    candidates_found: int
    warnings: list[str] = field(default_factory=list)
