"""Structured, redacted audit events for user-visible workflow actions."""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from services.persistence import Storage, utc_now

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {
    "api_key", "api_secret", "authorization", "password", "token",
    "card_number", "cvv", "expiry", "raw_card_data", "secret",
}
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in _SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _EMAIL.sub("[REDACTED_EMAIL]", value)[:4000]
    return value


class AuditLogger:
    def __init__(self, storage: Storage, correlation_id: str | None = None) -> None:
        self.storage = storage
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def record(self, event_type: str, *, profile_id: str | None = None,
               trip_id: str | None = None, data: Any = None) -> str | None:
        event = {
            "correlation_id": self.correlation_id,
            "event_type": event_type,
            "profile_id": profile_id,
            "trip_id": trip_id,
            "data": redact(data),
            "created_at": utc_now(),
        }
        try:
            return self.storage.save_audit_event(event)
        except Exception:
            logger.exception("Audit event could not be persisted: %s", event_type)
            return None
