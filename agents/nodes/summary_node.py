from models.booking import BookingRecord
from ui.console import console, print_trip_summary


def trip_summary_node(state: dict) -> dict:
    records = state.get("booking_records", [])
    total_items = len(state.get("bookable_items", []))
    skipped = total_items - len(records)
    trip_dict = state.get("trip_request", {})
    currency = trip_dict.get("preferred_currency", "USD")

    total_spent = 0.0
    for rec in records:
        details = rec.get("details", {}) if isinstance(rec, dict) else rec.details
        total_spent += float(
            details.get("total_price_usd") or details.get("price_usd") or 0
        )

    print_trip_summary(records, skipped, total_spent, currency)
    return {"phase": "done"}
