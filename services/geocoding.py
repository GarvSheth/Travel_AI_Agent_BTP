"""Place-name -> coordinates lookup via OpenStreetMap Nominatim.

Nuitee's /data/places returns place IDs without coordinates, so it cannot be
used for latitude/longitude. Nominatim is free and keyless; its usage policy
requires an identifying User-Agent and at most ~1 request/second, which a
single lookup per hotel search respects. Results are cached per process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import requests

from config.settings import GEOCODER_URL, GEOCODER_USER_AGENT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoLocation:
    latitude: float
    longitude: float
    display_name: str
    country_code: str | None = None     # ISO-2 upper-case, from Nominatim addressdetails


class GeocodingError(Exception):
    """Raised when the geocoding service cannot be reached or misbehaves."""


@lru_cache(maxsize=256)
def geocode(query: str, country_code: str | None = None) -> GeoLocation | None:
    """Resolve a free-text place to coordinates.

    Returns None when the place is not found. Raises GeocodingError on
    network or service errors. Coordinates always come from the service;
    nothing is guessed.
    """

    query = (query or "").strip()
    if not query:
        return None

    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}
    if country_code:
        params["countrycodes"] = country_code.strip().lower()

    try:
        response = requests.get(
            GEOCODER_URL,
            params=params,
            headers={"User-Agent": GEOCODER_USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise GeocodingError(f"Geocoding failed for '{query}': {exc}") from exc

    if not isinstance(results, list) or not results:
        logger.info("Geocoding found no match for '%s'.", query)
        return None

    top = results[0]
    try:
        location = GeoLocation(
            latitude=float(top["lat"]),
            longitude=float(top["lon"]),
            display_name=str(top.get("display_name") or query),
            country_code=(str((top.get("address") or {}).get("country_code") or "").upper() or None),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodingError(f"Unexpected geocoding response for '{query}'.") from exc

    logger.info(
        "Geocoded '%s' -> (%.5f, %.5f)", query, location.latitude, location.longitude
    )
    return location
