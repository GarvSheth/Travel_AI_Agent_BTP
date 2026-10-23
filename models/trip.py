from typing import Optional
from pydantic import BaseModel, Field


class HotelPreferences(BaseModel):
    star_rating_min: int = 1
    star_rating_max: int = 5
    max_price_per_night: Optional[float] = None
    min_rating: float = 0.0
    min_review_count: int = 0
    preferred_location: Optional[str] = None  # "city_center"|"near_attractions"|"near_beach"|"airport"|"any"
    required_amenities: list[str] = Field(default_factory=list)
    room_type: str = "any"  # "single"|"double"|"family"|"suite"|"any"


class FlightPreferences(BaseModel):
    max_price: Optional[float] = None
    max_stops: int = 2  # 0=direct, 1=one stop max, 2=any
    cabin_class: str = "economy"  # "economy"|"business"|"first"
    preferred_airlines: list[str] = Field(default_factory=list)
    preferred_departure_time: str = "any"  # "morning"|"afternoon"|"evening"|"any"
    max_layover_hours: Optional[float] = None


class TripRequest(BaseModel):
    destination: str
    origin: str
    departure_date: str   # "YYYY-MM-DD"
    return_date: str      # "YYYY-MM-DD"
    num_travelers: int = Field(ge=1, le=20)
    budget_usd: float = Field(gt=0)
    preferred_currency: str = "USD"
    accommodation_type: str = "any"  # "hotel"|"hostel"|"airbnb"|"any"
    trip_style: Optional[str] = None
    special_requirements: list[str] = Field(default_factory=list)
    hotel_preferences: HotelPreferences = Field(default_factory=HotelPreferences)
    flight_preferences: FlightPreferences = Field(default_factory=FlightPreferences)

    @property
    def num_nights(self) -> int:
        from datetime import datetime
        dep = datetime.strptime(self.departure_date, "%Y-%m-%d").date()
        ret = datetime.strptime(self.return_date, "%Y-%m-%d").date()
        return max((ret - dep).days, 1)


class Activity(BaseModel):
    name: str
    description: str
    estimated_cost_usd: Optional[float] = None
    duration_hours: Optional[float] = None
    bookable: bool = False


class FlightSegment(BaseModel):
    origin: str
    destination: str
    date: str  # "YYYY-MM-DD"
    duration_hours: Optional[float] = None


class HotelStay(BaseModel):
    city: str
    check_in: str   # "YYYY-MM-DD"
    check_out: str  # "YYYY-MM-DD"
    preferred_type: str = "any"


class Day(BaseModel):
    date: str  # "YYYY-MM-DD"
    location: str
    activities: list[Activity] = Field(default_factory=list)
    flight: Optional[FlightSegment] = None
    hotel: Optional[HotelStay] = None
    notes: Optional[str] = None


class Itinerary(BaseModel):
    trip_name: str
    destination: str
    total_days: int
    days: list[Day]
    estimated_total_cost_usd: float
    generated_at: Optional[str] = None
