"""Local profile identity and persistence helpers."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from services.audit_logger import AuditLogger
from services.persistence import Storage, create_storage


def profile_id() -> str:
    configured = os.getenv("TRAVEL_PROFILE_ID")
    if configured:
        return configured
    path = Path.home() / ".travel-agent" / "profile-id"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(str(uuid.uuid4()))
        path.chmod(0o600)
    return path.read_text().strip()


def save_profile(profile: dict, *, storage: Storage | None = None) -> str:
    storage = storage or create_storage()
    value = dict(profile)
    value["profile_id"] = profile_id()
    storage.save_profile(value)
    return value["profile_id"]


def audit(
    storage: Storage,
    event_type: str,
    *,
    trip_id: str | None = None,
    correlation_id: str | None = None,
    data=None,
) -> None:
    AuditLogger(
        storage,
        correlation_id=correlation_id or os.getenv("TRAVEL_CORRELATION_ID"),
    ).record(event_type, profile_id=profile_id(), trip_id=trip_id, data=data)
