"""Optional real-data adapters for destination enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class DestinationDataClient:
    places_api_key: str | None = None
    weather_api_key: str | None = None
    currency_api_key: str | None = None
    timeout: float = 10.0

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = httpx.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def weather(self, latitude: float, longitude: float) -> dict[str, Any]:
        if not self.weather_api_key:
            raise RuntimeError("WEATHER_API_KEY is required for weather data.")
        return self._get(
            "https://api.openweathermap.org/data/2.5/forecast",
            {"lat": latitude, "lon": longitude, "appid": self.weather_api_key, "units": "metric"},
        )

    def exchange_rates(self, base: str, target: str) -> dict[str, Any]:
        if not self.currency_api_key:
            raise RuntimeError("CURRENCY_API_KEY is required for currency data.")
        return self._get(
            "https://api.currencyapi.com/v3/latest",
            {"apikey": self.currency_api_key, "base_currency": base, "currencies": target},
        )

    def places(self, query: str) -> dict[str, Any]:
        if not self.places_api_key:
            raise RuntimeError("PLACES_API_KEY is required for destination data.")
        return self._get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            {"query": query, "key": self.places_api_key},
        )
