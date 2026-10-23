from models.booking import FlightOption, HotelOption, ScoredOption
from models.trip import FlightPreferences, HotelPreferences


def _score_label(score: int) -> str:
    if score >= 85:
        return "Excellent Match"
    if score >= 70:
        return "Great Match"
    if score >= 55:
        return "Good Match"
    if score >= 40:
        return "Partial Match"
    return "Poor Match"


def score_hotel(hotel: HotelOption, prefs: HotelPreferences) -> ScoredOption:
    breakdown: dict[str, int] = {}

    # --- Hard filters ---
    if hotel.stars < prefs.star_rating_min:
        return ScoredOption(option=hotel, score=0, label="Below min stars",
                            breakdown={"hard_filter": 0}, passes_hard_filters=False)
    if hotel.stars > prefs.star_rating_max:
        return ScoredOption(option=hotel, score=0, label="Above max stars",
                            breakdown={"hard_filter": 0}, passes_hard_filters=False)
    if prefs.max_price_per_night and hotel.price_per_night_usd > prefs.max_price_per_night:
        return ScoredOption(option=hotel, score=0, label="Over budget",
                            breakdown={"hard_filter": 0}, passes_hard_filters=False)

    score = 0.0

    # Rating (30 pts)
    if prefs.min_rating > 0 and hotel.rating is not None:
        pts = min(hotel.rating / prefs.min_rating, 1.2) * 30
        pts = min(pts, 30)
    else:
        pts = 30.0
    breakdown["rating"] = round(pts)
    score += pts

    # Review count (20 pts)
    if prefs.min_review_count > 0 and hotel.review_count is not None:
        pts = min(hotel.review_count / prefs.min_review_count, 2.0) * 20
        pts = min(pts, 20)
    else:
        pts = 20.0
    breakdown["reviews"] = round(pts)
    score += pts

    # Price efficiency (20 pts) — cheaper within budget scores higher
    if prefs.max_price_per_night and prefs.max_price_per_night > 0:
        pts = max((1 - hotel.price_per_night_usd / prefs.max_price_per_night) * 20, 0)
    else:
        pts = 10.0
    breakdown["price_eff"] = round(pts)
    score += pts

    # Location match (15 pts)
    if prefs.preferred_location and prefs.preferred_location != "any":
        pts = 15.0 if hotel.location_type == prefs.preferred_location else 0.0
    else:
        pts = 15.0
    breakdown["location"] = round(pts)
    score += pts

    # Amenity match (15 pts)
    if prefs.required_amenities:
        matched = sum(1 for a in prefs.required_amenities if a in hotel.amenities)
        pts = (matched / len(prefs.required_amenities)) * 15
    else:
        pts = 15.0
    breakdown["amenities"] = round(pts)
    score += pts

    final = min(round(score), 100)
    return ScoredOption(option=hotel, score=final, label=_score_label(final),
                        breakdown=breakdown, passes_hard_filters=True)


def score_flight(flight: FlightOption, prefs: FlightPreferences) -> ScoredOption:
    breakdown: dict[str, int] = {}

    # --- Hard filters ---
    if prefs.max_price and flight.price_usd > prefs.max_price:
        return ScoredOption(option=flight, score=0, label="Over budget",
                            breakdown={"hard_filter": 0}, passes_hard_filters=False)
    if flight.stops > prefs.max_stops:
        return ScoredOption(option=flight, score=0, label="Too many stops",
                            breakdown={"hard_filter": 0}, passes_hard_filters=False)

    score = 0.0

    # Price efficiency (40 pts)
    if prefs.max_price and prefs.max_price > 0:
        pts = max((1 - flight.price_usd / prefs.max_price) * 40, 0)
    else:
        pts = 20.0
    breakdown["price_eff"] = round(pts)
    score += pts

    # Stop count (30 pts)
    if prefs.max_stops > 0:
        pts = ((prefs.max_stops - flight.stops) / prefs.max_stops) * 30
    else:
        pts = 30.0  # direct required and this passed hard filter, so it is direct
    breakdown["stops"] = round(pts)
    score += pts

    # Departure time (20 pts)
    if prefs.preferred_departure_time != "any":
        pts = 20.0 if flight.departure_period == prefs.preferred_departure_time else 0.0
    else:
        pts = 20.0
    breakdown["departure"] = round(pts)
    score += pts

    # Airline preference (10 pts)
    if prefs.preferred_airlines:
        pts = 10.0 if flight.airline in prefs.preferred_airlines else 0.0
    else:
        pts = 10.0
    breakdown["airline"] = round(pts)
    score += pts

    final = min(round(score), 100)
    return ScoredOption(option=flight, score=final, label=_score_label(final),
                        breakdown=breakdown, passes_hard_filters=True)


def rank_options(
    options: list[FlightOption | HotelOption],
    prefs: FlightPreferences | HotelPreferences,
) -> list[ScoredOption]:
    """Score all options, exclude hard-filter failures, sort descending by score."""
    scored = []
    for opt in options:
        if isinstance(opt, HotelOption):
            assert isinstance(prefs, HotelPreferences)
            scored.append(score_hotel(opt, prefs))
        else:
            assert isinstance(prefs, FlightPreferences)
            scored.append(score_flight(opt, prefs))

    passing = [s for s in scored if s.passes_hard_filters]
    passing.sort(key=lambda s: s.score, reverse=True)
    return passing
