import random
import time
import uuid
import os
from typing import Optional

from models.booking import FlightOption, HotelOption


MOCK_AIRLINES = ["Air India", "IndiGo", "SpiceJet", "Vistara", "GoFirst", "Emirates", "Singapore Airlines"]
MOCK_HOTELS = [
    ("The Grand Residency", 5, 4.8, 2341, ["wifi", "pool", "gym", "breakfast", "parking", "ac"], "city_center"),
    ("Comfort Suites", 4, 4.6, 1123, ["wifi", "gym", "breakfast", "ac"], "near_attractions"),
    ("Heritage Inn", 3, 4.5, 892, ["wifi", "ac", "parking"], "near_attractions"),
    ("Budget Stay Plus", 3, 4.2, 412, ["wifi", "ac"], "city_center"),
    ("Boutique Palace", 4, 4.7, 678, ["wifi", "pool", "breakfast", "ac"], "near_beach"),
    ("Airport Transit Hotel", 3, 4.0, 234, ["wifi", "parking", "ac"], "airport"),
    ("City Center Lodge", 4, 4.4, 567, ["wifi", "gym", "ac"], "city_center"),
]


class TravelSearchClient:
    def __init__(self, api_key: str = "", api_secret: str = "",
                 hostname: str = "test"):
        self._live = False
        self._allow_mock = os.getenv("TRAVEL_DATA_MODE", "production").lower() == "mock"
        if api_key and api_secret:
            try:
                import amadeus
                self._client = amadeus.Client(
                    client_id=api_key,
                    client_secret=api_secret,
                    hostname=hostname,
                )
                self._live = True
            except Exception:
                self._live = False

    # ── Flights ──────────────────────────────────────────────────────────────

    def search_flights(self, origin: str, destination: str, date: str,
                       adults: int = 1, max_results: int = 5) -> list[FlightOption]:
        if self._live:
            try:
                return self._search_flights_live(origin, destination, date, adults, max_results)
            except Exception as exc:
                if not self._allow_mock:
                    raise RuntimeError(f"Amadeus flight search failed: {exc}") from exc
        if self._allow_mock:
            return self._mock_flights(origin, destination, date, max_results)
        raise RuntimeError("Amadeus credentials are not configured and mock data is disabled.")

    def _search_flights_live(self, origin: str, destination: str, date: str,
                              adults: int, max_results: int) -> list[FlightOption]:
        import amadeus
        resp = self._client.shopping.flight_offers_search.get(
            originLocationCode=origin.upper()[:3],
            destinationLocationCode=destination.upper()[:3],
            departureDate=date,
            adults=adults,
            max=max_results,
        )
        return [self._parse_flight(o) for o in resp.data[:max_results]]

    def _parse_flight(self, offer: dict) -> FlightOption:
        itin = offer["itineraries"][0]
        seg = itin["segments"][0]
        dep_time = seg["departure"]["at"]
        arr_time = seg["arrival"]["at"]
        hour = int(dep_time[11:13]) if len(dep_time) > 11 else 12
        if hour < 12:
            period = "morning"
        elif hour < 17:
            period = "afternoon"
        else:
            period = "evening"

        return FlightOption(
            option_id=offer.get("id", str(uuid.uuid4())),
            airline=seg.get("carrierCode", "XX"),
            flight_number=f"{seg.get('carrierCode','XX')}{seg.get('number','000')}",
            origin=seg["departure"]["iataCode"],
            destination=seg["arrival"]["iataCode"],
            departure_time=dep_time,
            arrival_time=arr_time,
            duration=itin.get("duration", "PT0H"),
            price_usd=float(offer["price"]["total"]),
            stops=len(itin["segments"]) - 1,
            departure_period=period,
            source="amadeus",
        )

    def _mock_flights(self, origin: str, destination: str, date: str,
                      max_results: int) -> list[FlightOption]:
        rng = random.Random(hash(origin + destination + date))
        results = []
        base_price = rng.uniform(150, 900)
        for i in range(min(max_results, 5)):
            airline = rng.choice(MOCK_AIRLINES)
            stops = rng.choice([0, 0, 0, 1, 1, 2])
            price = round(base_price * (1 + i * 0.12) * (1 + stops * 0.05), 2)
            dep_hour = rng.choice([6, 8, 10, 14, 18, 20])
            period = "morning" if dep_hour < 12 else ("afternoon" if dep_hour < 17 else "evening")
            results.append(FlightOption(
                option_id=f"mock-fl-{i}",
                airline=airline,
                flight_number=f"{airline[:2].upper()}{rng.randint(100, 999)}",
                origin=origin.upper()[:3],
                destination=destination.upper()[:3],
                departure_time=f"{date}T{dep_hour:02d}:00:00",
                arrival_time=f"{date}T{(dep_hour + 2 + stops * 2) % 24:02d}:30:00",
                duration=f"PT{2 + stops * 2}H30M",
                price_usd=price,
                stops=stops,
                departure_period=period,
                source="mock",
            ))
        return results

    # ── Hotels ───────────────────────────────────────────────────────────────

    def search_hotels(self, city_code: str, check_in: str, check_out: str,
                      adults: int = 1, max_results: int = 5) -> list[HotelOption]:
        if self._live:
            try:
                return self._search_hotels_live(city_code, check_in, check_out, adults, max_results)
            except Exception as exc:
                if not self._allow_mock:
                    raise RuntimeError(f"Amadeus hotel search failed: {exc}") from exc
        if self._allow_mock:
            return self._mock_hotels(city_code, check_in, check_out, max_results)
        raise RuntimeError("Amadeus credentials are not configured and mock data is disabled.")

    def _search_hotels_live(self, city_code: str, check_in: str, check_out: str,
                             adults: int, max_results: int) -> list[HotelOption]:
        # Step 1: find hotel IDs by city
        time.sleep(0.5)  # Amadeus free tier rate limit
        hotels_resp = self._client.reference_data.locations.hotels.by_city.get(
            cityCode=city_code.upper()[:3]
        )
        hotel_ids = [h["hotelId"] for h in hotels_resp.data[:20]]
        if not hotel_ids:
            if self._allow_mock:
                return self._mock_hotels(city_code, check_in, check_out, max_results)
            raise RuntimeError("Amadeus returned no hotels for the requested city.")

        # Step 2: get offers for those hotels
        time.sleep(0.5)
        offers_resp = self._client.shopping.hotel_offers_search.get(
            hotelIds=hotel_ids[:10],
            checkInDate=check_in,
            checkOutDate=check_out,
            adults=adults,
        )
        return [self._parse_hotel(o, check_in, check_out) for o in offers_resp.data[:max_results]]

    def _parse_hotel(self, offer: dict, check_in: str, check_out: str) -> HotelOption:
        hotel = offer.get("hotel", {})
        price_info = offer.get("offers", [{}])[0].get("price", {})
        try:
            nights = max((
                __import__("datetime").datetime.strptime(check_out, "%Y-%m-%d") -
                __import__("datetime").datetime.strptime(check_in, "%Y-%m-%d")
            ).days, 1)
        except Exception:
            nights = 1
        total = float(price_info.get("total", 0))
        per_night = round(total / nights, 2)

        return HotelOption(
            option_id=hotel.get("hotelId", str(uuid.uuid4())),
            name=hotel.get("name", "Unknown Hotel"),
            city=hotel.get("cityCode", ""),
            stars=int(hotel.get("rating", 3)),
            rating=None,
            review_count=None,
            location_type=None,
            amenities=[a.lower() for a in hotel.get("amenities", [])[:8]],
            price_per_night_usd=per_night,
            total_price_usd=total,
            check_in=check_in,
            check_out=check_out,
            source="amadeus",
        )

    def _mock_hotels(self, city_code: str, check_in: str, check_out: str,
                     max_results: int) -> list[HotelOption]:
        rng = random.Random(hash(city_code + check_in))
        try:
            from datetime import datetime
            nights = max((datetime.strptime(check_out, "%Y-%m-%d") -
                          datetime.strptime(check_in, "%Y-%m-%d")).days, 1)
        except Exception:
            nights = 3

        shuffled = list(MOCK_HOTELS)
        rng.shuffle(shuffled)
        results = []
        for i, (name, stars, rating, reviews, amenities, loc) in enumerate(shuffled[:max_results]):
            base = rng.uniform(40, 250)
            per_night = round(base * (1 + (stars - 3) * 0.3) * (1 + i * 0.05), 2)
            results.append(HotelOption(
                option_id=f"mock-ht-{i}",
                name=name,
                city=city_code.upper()[:3],
                stars=stars,
                rating=rating,
                review_count=reviews + rng.randint(-50, 50),
                location_type=loc,
                amenities=amenities,
                price_per_night_usd=per_night,
                total_price_usd=round(per_night * nights, 2),
                check_in=check_in,
                check_out=check_out,
                source="mock",
            ))
        return results
