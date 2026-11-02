"""Duffel API client for real-time flight offer search.

Endpoints used:
    POST /air/offer_requests   create a search and return matching offers

Duffel API reference: https://duffel.com/docs/api/v2/offer-requests
Authentication: Bearer token in Authorization header, Duffel-Version: v2
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_URL = "https://api.duffel.com"
_VALID_CABIN_CLASSES = {"economy", "premium_economy", "business", "first"}


class DuffelAuthError(Exception):
    """Raised when DUFFEL_API_KEY is missing from the environment."""


class DuffelAPIError(Exception):
    """Raised when the Duffel API returns a non-2xx response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class DuffelClient:
    """Reusable HTTP client for the Duffel Flights API."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("DUFFEL_API_KEY")
        if not api_key:
            raise DuffelAuthError(
                "DUFFEL_API_KEY is missing from the environment or .env file."
            )
        self._base_url = BASE_URL
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Duffel-Version": "v2",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        num_passengers: int = 1,
        cabin_class: str | None = "economy",
        max_connections: int | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Search for flight offers via POST /air/offer_requests.

        Returns up to ``limit`` raw offer dicts from the Duffel response.
        Raises DuffelAPIError on search failure, DuffelAuthError if no key.
        """
        if not origin or not destination or not departure_date:
            raise ValueError("origin, destination, and departure_date are required.")
        if num_passengers < 1:
            raise ValueError("At least one passenger is required.")

        payload: dict[str, Any] = {
            "data": {
                "slices": [
                    {
                        "origin": origin.strip().upper(),
                        "destination": destination.strip().upper(),
                        "departure_date": departure_date,
                    }
                ],
                "passengers": [{"type": "adult"} for _ in range(num_passengers)],
            }
        }
        if cabin_class and cabin_class.lower() in _VALID_CABIN_CLASSES:
            payload["data"]["cabin_class"] = cabin_class.lower()
        if max_connections is not None and max_connections >= 0:
            payload["data"]["max_connections"] = max_connections

        logger.info(
            "Duffel search: %s → %s on %s (%d pax, cabin=%s, max_conn=%s)",
            origin, destination, departure_date, num_passengers,
            cabin_class, max_connections,
        )

        response = self._request(
            "POST",
            "/air/offer_requests",
            params={"return_offers": "true"},
            payload=payload,
            timeout=45,
        )
        offers = (response.get("offers") or [])
        logger.info("Duffel returned %d offers (returning up to %d).", len(offers), limit)
        return offers[:limit]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        try:
            resp = self._session.request(
                method, url, params=params, json=payload, timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise DuffelAPIError("Duffel request timed out.") from exc
        except requests.exceptions.RequestException as exc:
            raise DuffelAPIError(f"Network error: {type(exc).__name__}") from exc
        return self._parse_response(resp)

    @staticmethod
    def _parse_response(response: requests.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}

        if not response.ok:
            errors = body.get("errors") or []
            msg = errors[0].get("message") if errors else f"HTTP {response.status_code}"
            raise DuffelAPIError(
                f"Duffel API error: {msg}",
                status_code=response.status_code,
                response_data=body,
            )

        data = body.get("data")
        if not isinstance(data, dict):
            raise DuffelAPIError(
                "Unexpected Duffel response format.",
                status_code=response.status_code,
                response_data=body,
            )
        return data

    def close(self) -> None:
        self._session.close()


# ── Normalizer (module-level, no API calls) ───────────────────────────────────

def _period(iso_datetime: str) -> str:
    """Return 'morning' / 'afternoon' / 'evening' from an ISO 8601 timestamp."""
    try:
        hour = int(iso_datetime[11:13])
    except (IndexError, ValueError):
        return "any"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def normalize_offer(offer: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Duffel offer into a flat dict for the flight agent state.

    Fields mirror the shape of models.booking.FlightOption plus extras
    (currency, cabin_class) needed for ranking.
    """
    slice_ = (offer.get("slices") or [{}])[0]
    segments = slice_.get("segments") or [{}]
    first_seg = segments[0]
    last_seg = segments[-1]

    carrier = first_seg.get("marketing_carrier") or {}
    carrier_code = carrier.get("iata_code") or ""
    flight_num = first_seg.get("marketing_carrier_flight_number") or ""

    seg_pax = first_seg.get("passengers") or []
    cabin = seg_pax[0].get("cabin_class") if seg_pax else (offer.get("cabin_class") or "economy")

    return {
        "option_id": offer.get("id") or "",
        "airline": (offer.get("owner") or {}).get("name") or carrier_code,
        "airline_iata": (offer.get("owner") or {}).get("iata_code") or carrier_code,
        "flight_number": f"{carrier_code}{flight_num}".strip() or "N/A",
        "origin": (first_seg.get("origin") or {}).get("iata_code") or "",
        "destination": (last_seg.get("destination") or {}).get("iata_code") or "",
        "departure_time": first_seg.get("departing_at") or "",
        "arrival_time": last_seg.get("arriving_at") or "",
        "duration": slice_.get("duration") or "",
        "price_usd": float(offer.get("total_amount") or 0),
        "currency": offer.get("total_currency") or "USD",
        "stops": max(len(segments) - 1, 0),
        "departure_period": _period(first_seg.get("departing_at") or ""),
        "cabin_class": cabin or "economy",
        "source": "duffel",
    }
