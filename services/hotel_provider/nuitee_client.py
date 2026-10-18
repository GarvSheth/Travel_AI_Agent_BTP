
"""Nuitee Connect (LiteAPI) client for hotel data, search and availability.

Endpoints used (all verified against docs.liteapi.travel and live responses):
    GET  /data/hotels       hotel metadata search (location, stars, rating)
    GET  /data/facilities   facility id -> name lookup
    GET  /data/hotel/ask    conversational Q&A about a single hotel (beta)
    POST /hotels/rates      live rates and availability
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

BASE_URL = "https://api.liteapi.travel/v3.0"
_VALID_STARS = {x / 2 for x in range(2, 11)}  # 1.0, 1.5, ... 5.0


class NuiteeAuthError(Exception):
    """Raised when Nuitee API credentials are missing."""


class NuiteeAPIError(Exception):
    """Raised when the Nuitee API returns an unsuccessful response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class NuiteeClient:
    """Reusable client for Nuitee Connect hotel APIs."""

    def __init__(self) -> None:
        load_dotenv()

        self.api_key = os.getenv("NUITEE_API_KEY")

        if not self.api_key:
            raise NuiteeAuthError(
                "NUITEE_API_KEY is missing from the environment or .env file."
            )

        self.base_url = BASE_URL
        self._facilities: dict[int, str] | None = None

        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": self.api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _validate_search(
        check_in: str,
        check_out: str,
        adults: int,
        rooms: int,
    ) -> None:
        try:
            checkin_date = date.fromisoformat(check_in)
            checkout_date = date.fromisoformat(check_out)
        except ValueError as exc:
            raise ValueError("Dates must use YYYY-MM-DD format.") from exc

        if checkout_date <= checkin_date:
            raise ValueError("Check-out must be after check-in.")

        if adults < 1:
            raise ValueError("At least one adult is required.")

        if rooms < 1:
            raise ValueError("At least one room is required.")

        if adults < rooms:
            raise ValueError("Adult count must be at least equal to room count.")

    @staticmethod
    def _parse_response(response: requests.Response) -> dict[str, Any]:
        try:
            response_data = response.json()
        except ValueError:
            response_data = {"raw_response": response.text}

        if not response.ok:
            raise NuiteeAPIError(
                message=f"Nuitee API returned HTTP {response.status_code}.",
                status_code=response.status_code,
                response_data=response_data,
            )

        if not isinstance(response_data, dict):
            raise NuiteeAPIError(
                "Unexpected Nuitee response format.",
                status_code=response.status_code,
                response_data=response_data,
            )

        return response_data

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Send a request and return parsed JSON. The API key is never logged."""

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise NuiteeAPIError("Nuitee request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            raise NuiteeAPIError(f"Network error: {type(exc).__name__}") from exc

        return self._parse_response(response)

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a POST request and return parsed JSON."""
        return self._request("POST", endpoint, payload=payload)

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any],
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Send a GET request and return parsed JSON."""
        return self._request("GET", endpoint, params=params, timeout=timeout)

    # ── Hotel data ───────────────────────────────────────────────────────

    def search_hotels(
        self,
        country_code: str,
        city_name: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius: int = 5000,
        star_rating: list[float] | None = None,
        min_rating: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search hotel metadata via GET /data/hotels.

        Returns the raw hotel objects from the ``data`` array (possibly empty).
        ``radius`` (metres, API minimum 1000) is only sent with coordinates.
        """

        if not country_code or not country_code.strip():
            raise ValueError("Country code cannot be empty.")

        if (latitude is None) != (longitude is None):
            raise ValueError("Latitude and longitude must be provided together.")

        if latitude is None and not (city_name and city_name.strip()):
            raise ValueError("Provide a city name or coordinates.")

        if not 1 <= limit <= 5000:
            raise ValueError("Limit must be between 1 and 5000.")

        params: dict[str, Any] = {
            "countryCode": country_code.strip().upper(),
            "limit": limit,
        }

        if city_name and city_name.strip():
            params["cityName"] = city_name.strip()

        if latitude is not None:
            if radius < 1000:
                raise ValueError("Radius must be at least 1000 metres.")
            params.update(
                {"latitude": latitude, "longitude": longitude, "radius": radius}
            )

        if star_rating:
            stars = sorted({float(s) for s in star_rating})
            invalid = [s for s in stars if s not in _VALID_STARS]
            if invalid:
                raise ValueError(f"Invalid star ratings: {invalid}.")
            # API expects e.g. "3.5,4.0,5.0"
            params["starRating"] = ",".join(f"{s:.1f}" for s in stars)

        if min_rating is not None:
            if not 0 <= min_rating <= 10:
                raise ValueError("Minimum rating must be between 0 and 10.")
            params["minRating"] = min_rating

        logger.info(
            "Nuitee /data/hotels search: %s",
            params,
        )

        response = self._get("/data/hotels", params)
        hotels = response.get("data") or []

        if not isinstance(hotels, list):
            raise NuiteeAPIError(
                "Unexpected /data/hotels response format.",
                response_data=response,
            )

        logger.info(
            "Nuitee /data/hotels returned %d hotels (total=%s).",
            len(hotels),
            response.get("total"),
        )
        return [h for h in hotels if isinstance(h, dict) and h.get("id")]

    def get_facilities(self) -> dict[int, str]:
        """Return the facility id -> English name map (cached per client)."""

        if self._facilities is None:
            response = self._get("/data/facilities", {})
            self._facilities = {
                int(item["facility_id"]): str(item["facility"])
                for item in response.get("data") or []
                if isinstance(item, dict)
                and item.get("facility_id") is not None
                and item.get("facility")
            }
        return self._facilities

    def get_hotel_details(self, hotel_id: str, query: str) -> str:
        """Ask a question about one hotel via GET /data/hotel/ask (beta).

        Returns the answer text. Raises NuiteeAPIError on failure or an
        empty answer. Web search is left disabled (API default) so answers
        come from Nuitee's own hotel information.
        """

        if not hotel_id or not hotel_id.strip():
            raise ValueError("Hotel ID cannot be empty.")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        response = self._get(
            "/data/hotel/ask",
            {"hotelId": hotel_id.strip(), "query": query.strip()},
            timeout=60,
        )
        data = response.get("data")
        answer = data.get("answer") if isinstance(data, dict) else None

        if not isinstance(answer, str) or not answer.strip():
            raise NuiteeAPIError(
                "Nuitee /data/hotel/ask returned no answer.",
                response_data=response,
            )

        return answer.strip()

    # ── Rates ────────────────────────────────────────────────────────────

    def search_hotel_rates(
        self,
        city: str,
        country_code: str,
        check_in: str,
        check_out: str,
        adults: int = 2,
        rooms: int = 1,
        currency: str = "INR",
        guest_nationality: str = "IN",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for hotels and available rates in a city."""

        if not city.strip():
            raise ValueError("City name cannot be empty.")

        if not country_code.strip():
            raise ValueError("Country code cannot be empty.")

        if not currency.strip():
            raise ValueError("Currency cannot be empty.")

        if not guest_nationality.strip():
            raise ValueError("Guest nationality cannot be empty.")

        if not 1 <= limit <= 5000:
            raise ValueError("Limit must be between 1 and 5000.")

        self._validate_search(check_in, check_out, adults, rooms)

        payload = {
            "cityName": city,
            "countryCode": country_code,
            "checkin": check_in,
            "checkout": check_out,
            "occupancies": [
                {"adults": adults}
                for _ in range(rooms)
            ],
            "currency": currency,
            "guestNationality": guest_nationality,
            "limit": limit,
            "timeout": 10,
            "roomMapping": True,
        }

        return self._post("/hotels/rates", payload)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()
