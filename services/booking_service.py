"""Booking boundary. Real checkout must be implemented by a provider adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BookingProvider(Protocol):
    def create_checkout(self, offer: dict[str, Any], *, idempotency_key: str) -> str: ...


class BookingUnavailableError(RuntimeError):
    pass


@dataclass
class HostedCheckoutProvider:
    """Contract for a PCI-compliant hosted checkout integration."""

    name: str

    def create_checkout(self, offer: dict[str, Any], *, idempotency_key: str) -> str:
        raise BookingUnavailableError(
            f"{self.name} checkout is not configured. "
            "Configure a hosted/tokenized payment provider before booking."
        )


def create_booking_provider() -> BookingProvider:
    return HostedCheckoutProvider(name="Configured travel provider")
