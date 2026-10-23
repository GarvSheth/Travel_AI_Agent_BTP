"""Durable application storage with MongoDB and a test-friendly in-memory adapter."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Protocol


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Storage(Protocol):
    def save_profile(self, profile: dict[str, Any]) -> str: ...
    def get_profile(self, profile_id: str) -> Optional[dict[str, Any]]: ...
    def save_trip(self, trip: dict[str, Any]) -> str: ...
    def save_audit_event(self, event: dict[str, Any]) -> str: ...


class InMemoryStorage:
    """Small adapter for tests and local development without MongoDB."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict[str, Any]] = {}
        self.trips: dict[str, dict[str, Any]] = {}
        self.audit_events: list[dict[str, Any]] = []

    def save_profile(self, profile: dict[str, Any]) -> str:
        value = dict(profile)
        profile_id = value.setdefault("profile_id", str(uuid.uuid4()))
        value["updated_at"] = utc_now()
        self.profiles[profile_id] = value
        return profile_id

    def get_profile(self, profile_id: str) -> Optional[dict[str, Any]]:
        return self.profiles.get(profile_id)

    def save_trip(self, trip: dict[str, Any]) -> str:
        value = dict(trip)
        trip_id = value.setdefault("trip_id", str(uuid.uuid4()))
        value["updated_at"] = utc_now()
        self.trips[trip_id] = value
        return trip_id

    def save_audit_event(self, event: dict[str, Any]) -> str:
        value = dict(event)
        value.setdefault("event_id", str(uuid.uuid4()))
        value.setdefault("created_at", utc_now())
        self.audit_events.append(value)
        return value["event_id"]


class MongoStorage:
    """MongoDB adapter. Importing this module does not require pymongo."""

    def __init__(self, uri: str, database: str = "travel_agent") -> None:
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError("MongoDB storage requires the 'pymongo' package.") from exc

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[database]
        self.profiles = self.db["profiles"]
        self.trips = self.db["trips"]
        self.audit_events = self.db["audit_events"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.profiles.create_index("profile_id", unique=True)
        self.trips.create_index("trip_id", unique=True)
        self.trips.create_index([("profile_id", 1), ("updated_at", -1)])
        self.audit_events.create_index([("correlation_id", 1), ("created_at", 1)])
        self.audit_events.create_index("created_at", expireAfterSeconds=_retention_seconds())

    def save_profile(self, profile: dict[str, Any]) -> str:
        value = dict(profile)
        profile_id = value.setdefault("profile_id", str(uuid.uuid4()))
        value["updated_at"] = utc_now()
        self.profiles.replace_one({"profile_id": profile_id}, value, upsert=True)
        return profile_id

    def get_profile(self, profile_id: str) -> Optional[dict[str, Any]]:
        return self.profiles.find_one({"profile_id": profile_id}, {"_id": 0})

    def save_trip(self, trip: dict[str, Any]) -> str:
        value = dict(trip)
        trip_id = value.setdefault("trip_id", str(uuid.uuid4()))
        value["updated_at"] = utc_now()
        self.trips.replace_one({"trip_id": trip_id}, value, upsert=True)
        return trip_id

    def save_audit_event(self, event: dict[str, Any]) -> str:
        value = dict(event)
        value.setdefault("event_id", str(uuid.uuid4()))
        value.setdefault("created_at", utc_now())
        self.audit_events.insert_one(value)
        return value["event_id"]


def _retention_seconds() -> int:
    days = int(os.getenv("AUDIT_RETENTION_DAYS", "365"))
    if days < 1:
        raise ValueError("AUDIT_RETENTION_DAYS must be at least 1.")
    return days * 24 * 60 * 60


def create_storage(uri: str | None = None) -> Storage:
    uri = uri or os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is required for durable storage. "
            "Use InMemoryStorage explicitly in tests."
        )
    return MongoStorage(uri, os.getenv("MONGODB_DATABASE", "travel_agent"))
