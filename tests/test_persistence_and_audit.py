from services.audit_logger import AuditLogger, redact
from services.amadeus_client import TravelSearchClient
from services.booking_service import BookingUnavailableError, HostedCheckoutProvider
from services.persistence import InMemoryStorage


def test_redact_removes_secrets_and_email_addresses():
    value = redact({"api_key": "secret", "email": "user@example.com", "nested": {"cvv": "123"}})
    assert value == {
        "api_key": "[REDACTED]",
        "email": "[REDACTED_EMAIL]",
        "nested": {"cvv": "[REDACTED]"},
    }


def test_in_memory_storage_persists_profile_trip_and_audit_event():
    storage = InMemoryStorage()
    profile_id = storage.save_profile({"name": "Local traveler"})
    trip_id = storage.save_trip({"profile_id": profile_id, "destination": "Paris"})
    event_id = AuditLogger(storage, "corr-1").record(
        "trip.created", profile_id=profile_id, trip_id=trip_id,
        data={"card_number": "4111111111111111"},
    )

    assert storage.get_profile(profile_id)["name"] == "Local traveler"
    assert storage.trips[trip_id]["destination"] == "Paris"
    assert event_id is not None
    assert storage.audit_events[0]["data"]["card_number"] == "[REDACTED]"


def test_production_search_does_not_fall_back_to_mock(monkeypatch):
    monkeypatch.setenv("TRAVEL_DATA_MODE", "production")
    client = TravelSearchClient()
    try:
        client.search_flights("DEL", "LHR", "2030-01-01")
    except RuntimeError as exc:
        assert "mock data is disabled" in str(exc)
    else:
        raise AssertionError("production search unexpectedly returned mock data")


def test_checkout_requires_configured_provider():
    try:
        HostedCheckoutProvider("test").create_checkout({}, idempotency_key="one")
    except BookingUnavailableError as exc:
        assert "hosted/tokenized" in str(exc)
    else:
        raise AssertionError("unconfigured checkout unexpectedly succeeded")
