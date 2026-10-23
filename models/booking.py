from typing import Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class FlightOption(BaseModel):
    option_id: str
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    duration: str
    price_usd: float
    stops: int = 0
    departure_period: str = "any"  # "morning"|"afternoon"|"evening"
    source: str = "amadeus"


class HotelOption(BaseModel):
    option_id: str
    name: str
    city: str
    stars: int
    rating: Optional[float] = None
    review_count: Optional[int] = None
    location_type: Optional[str] = None  # "city_center"|"near_attractions"|"near_beach"|"airport"
    amenities: list[str] = Field(default_factory=list)
    price_per_night_usd: float
    total_price_usd: float
    check_in: str
    check_out: str
    source: str = "amadeus"


class ScoredOption(BaseModel):
    option: Union[FlightOption, HotelOption]
    score: int           # 0–100
    label: str           # "Excellent Match" etc.
    breakdown: dict[str, int]
    passes_hard_filters: bool


class BookingRecord(BaseModel):
    item_type: str
    confirmation_number: str
    details: dict[str, Any]
    status: str = "simulated"
    booked_at: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.booked_at:
            self.booked_at = datetime.now().isoformat()


class PaymentRequest(BaseModel):
    booking_type: str
    provider_name: str
    amount: float
    currency: str
    traveler_names: list[str]
    booking_reference: str
    simulate: bool = True


class PaymentResult(BaseModel):
    status: str   # "confirmed"|"failed"|"cancelled"
    confirmation_id: str
    amount_charged: float
    error_message: Optional[str] = None
