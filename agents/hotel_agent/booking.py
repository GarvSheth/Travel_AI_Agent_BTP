"""Prebook, book and retrieve a booking (Nuitee /rates/prebook, /rates/book, /bookings/{id}).

Flow: room offer (offerId) -> prebook -> PrebookResult (prebookId, confirmed price)
      -> guest details -> book -> BookingResult (bookingId, status, raw response)
      -> get_booking_details(bookingId) -> BookingDetails (confirmation + PDF).
Only values present in the API responses are used; missing ones stay None / empty.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

from services.hotel_provider.nuitee_client import NuiteeAPIError, NuiteeClient


@dataclass
class GuestDetails:
    first_name: str
    last_name: str
    email: str
    phone: str

    def as_holder(self) -> dict[str, str]:
        return {"firstName": self.first_name.strip(), "lastName": self.last_name.strip(),
                "email": self.email.strip(), "phone": self.phone.strip()}


@dataclass
class PrebookResult:
    prebook_id: str
    hotel_id: str
    hotel_name: str
    room_name: str
    board: str | None
    check_in: str
    check_out: str
    price: float
    currency: str
    refundable: bool | None
    rooms: int                              # occupancies in the offer (one guest each)
    price_changed: bool                     # priceDifferencePercent != 0
    terms_changed: bool                     # cancellation or board changed since search
    sandbox: bool = False


@dataclass
class BookingResult:
    booking_id: str
    status: str                             # "CONFIRMED" | "CANCELED" | ...
    hotel_name: str
    check_in: str
    check_out: str
    price: float | None
    currency: str
    hotel_confirmation_code: str | None
    refundable: bool | None
    last_free_cancellation: str | None
    sandbox: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)   # full /rates/book response


@dataclass
class BookedRoom:
    room_type: str | None
    board: str | None
    adults: int | None
    children: int | None
    guests: list[str]                       # "First Last" per guest


@dataclass
class BookingDetails:
    """A booking as stored by Nuitee (GET /bookings/{id}): source for the confirmation."""
    booking_id: str
    status: str
    sandbox: bool
    hotel_name: str | None
    hotel_address: str | None
    hotel_phone: str | None
    check_in: str | None
    check_out: str | None
    check_in_time: str | None
    check_out_time: str | None
    adults: int | None
    children: int | None
    rooms: list[BookedRoom]
    holder_name: str | None
    holder_email: str | None
    holder_phone: str | None
    total: float | None
    currency: str | None
    payment_status: str | None
    hotel_confirmation_code: str | None
    refundable: bool | None
    last_free_cancellation: str | None
    cancellation_rules: list[str]           # human-readable lines from cancelPolicyInfos
    remarks: list[str]                      # hotel / special remarks, know-before-you-go
    created_at: str | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.status.upper() == "CONFIRMED"

    @property
    def guest_count(self) -> int | None:
        if self.adults is None and self.children is None:
            return None
        return (self.adults or 0) + (self.children or 0)


def _refundable(tag: Any) -> bool | None:
    return {"RFN": True, "NRFN": False}.get(str(tag or "").upper())


def parse_prebook(response: dict[str, Any], hotel_name: str) -> PrebookResult:
    data = response.get("data")
    if not isinstance(data, dict) or not data.get("prebookId"):
        raise NuiteeAPIError("Prebook returned no prebookId.", response_data=response)

    rates = [r for rt in data.get("roomTypes") or [] if isinstance(rt, dict)
             for r in rt.get("rates") or [] if isinstance(r, dict)]
    first = rates[0] if rates else {}
    try:
        diff = float(data.get("priceDifferencePercent") or 0)
    except (TypeError, ValueError):
        diff = 0.0

    return PrebookResult(
        prebook_id=str(data["prebookId"]),
        hotel_id=str(data.get("hotelId") or ""),
        hotel_name=hotel_name,
        room_name=str(first.get("name") or "Room"),
        board=first.get("boardName") or None,
        check_in=str(data.get("checkin") or ""),
        check_out=str(data.get("checkout") or ""),
        price=float(data.get("price") or 0),
        currency=str(data.get("currency") or ""),
        refundable=_refundable((first.get("cancellationPolicies") or {}).get("refundableTag")),
        rooms=max(1, len({r.get("occupancyNumber") for r in rates}) if rates else 1),
        price_changed=diff != 0,
        terms_changed=bool(data.get("cancellationChanged") or data.get("boardChanged")),
        sandbox=bool(response.get("sandbox")),
    )


def booking_guests(guest: GuestDetails, rooms: int) -> list[dict[str, Any]]:
    """One primary guest per room (occupancyNumber 1..rooms), all the booking holder."""
    holder = guest.as_holder()
    return [{"occupancyNumber": n, "firstName": holder["firstName"],
             "lastName": holder["lastName"], "email": holder["email"]}
            for n in range(1, rooms + 1)]


def parse_booking(response: dict[str, Any], hotel_name: str) -> BookingResult:
    data = response.get("data")
    if not isinstance(data, dict) or not data.get("bookingId"):
        raise NuiteeAPIError("Booking returned no bookingId.", response_data=response)

    hotel = data.get("hotel") if isinstance(data.get("hotel"), dict) else {}
    price = data.get("price")
    return BookingResult(
        booking_id=str(data["bookingId"]),
        status=str(data.get("status") or "UNKNOWN"),
        hotel_name=str(hotel.get("name") or data.get("hotelName") or hotel_name),
        check_in=str(data.get("checkin") or ""),
        check_out=str(data.get("checkout") or ""),
        price=float(price) if isinstance(price, (int, float)) else None,
        currency=str(data.get("currency") or ""),
        hotel_confirmation_code=data.get("hotelConfirmationCode") or None,
        refundable=_refundable(data.get("tag")),
        last_free_cancellation=data.get("lastFreeCancellationDate") or None,
        sandbox=bool(response.get("sandbox") or data.get("sandbox")),
        raw=response,
    )


# ── Booking details (GET /bookings/{bookingId}) ──────────────────────────────

def _text(value: Any) -> str | None:
    """Non-empty stripped string, else None (the API uses "" for missing values)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def html_to_paragraphs(value: Any) -> list[str]:
    """Split HTML-ish remarks (<br/>, <p>, <li>) into plain-text paragraphs."""
    if isinstance(value, list):
        return [p for v in value for p in html_to_paragraphs(v)]
    text = _text(value)
    if not text:
        return []
    text = re.sub(r"(?i)<\s*(br|/p|/li|/div)\s*/?>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return [re.sub(r"\s+", " ", line).strip() for line in text.split("\n") if line.strip()]


def _cancellation_rules(policies: Any, currency: str | None) -> list[str]:
    infos = policies.get("cancelPolicyInfos") if isinstance(policies, dict) else None
    rules = []
    for info in infos or []:
        if not isinstance(info, dict):
            continue
        when = _text(info.get("cancelTime"))
        amount = info.get("amount")
        cur = _text(info.get("currency")) or currency or ""
        if when and isinstance(amount, (int, float)):
            tz = _text(info.get("timezone"))
            fee = "no fee" if amount == 0 else f"fee {amount:,.2f} {cur}".strip()
            rules.append(f"Cancelling from {when}{' ' + tz if tz else ''}: {fee}")
    return rules


def parse_booking_details(response: dict[str, Any]) -> BookingDetails:
    """Parse GET /bookings/{id} (also accepts the /rates/book response, same shape)."""
    data = response.get("data")
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or not data.get("bookingId"):
        raise NuiteeAPIError("Booking details have no bookingId.", response_data=response)

    hotel = data.get("hotel") if isinstance(data.get("hotel"), dict) else {}
    times = hotel.get("checkinCheckoutTimes") if isinstance(hotel.get("checkinCheckoutTimes"), dict) else {}
    holder = data.get("holder") if isinstance(data.get("holder"), dict) else {}
    currency = _text(data.get("currency"))

    rooms = []
    for room in data.get("bookedRooms") or []:
        if not isinstance(room, dict):
            continue
        room_type = room.get("roomType") if isinstance(room.get("roomType"), dict) else {}
        names = [" ".join(filter(None, (_text(g.get("firstName")), _text(g.get("lastName")))))
                 for g in room.get("guests") or [] if isinstance(g, dict)]
        if not any(names):
            names = [" ".join(filter(None, (_text(room.get("firstName")), _text(room.get("lastName")))))]
        rooms.append(BookedRoom(
            room_type=_text(room_type.get("name")),
            board=_text(room.get("boardName")),
            adults=_int(room.get("adults")),
            children=_int(room.get("children")),
            guests=[n for n in names if n],
        ))

    address = ", ".join(filter(None, (_text(hotel.get("address")), _text(hotel.get("city")),
                                      _text(hotel.get("zip")), _text(hotel.get("country")))))
    policies = data.get("cancellationPolicies")
    tag = data.get("tag") or (policies.get("refundableTag") if isinstance(policies, dict) else None)
    remarks = (html_to_paragraphs(data.get("hotelRemarks"))
               + html_to_paragraphs(data.get("specialRemarks"))
               + html_to_paragraphs(data.get("knowBeforeYouGo"))
               + html_to_paragraphs(data.get("remarks"))
               + html_to_paragraphs(times.get("special_instructions"))
               + html_to_paragraphs(times.get("instructions"))
               + (html_to_paragraphs(policies.get("hotelRemarks")) if isinstance(policies, dict) else []))
    price = data.get("price")
    adults = _int(data.get("adults"))
    if adults is None and rooms:
        adults = sum(r.adults or 0 for r in rooms) or None

    return BookingDetails(
        booking_id=str(data["bookingId"]),
        status=_text(data.get("status")) or "UNKNOWN",
        sandbox=bool(response.get("sandbox") or data.get("sandbox")),
        hotel_name=_text(hotel.get("name")) or _text(data.get("hotelName")),
        hotel_address=address or None,
        hotel_phone=_text(hotel.get("phone")),
        check_in=_text(data.get("checkin")),
        check_out=_text(data.get("checkout")),
        check_in_time=_text(times.get("checkin_start")),
        check_out_time=_text(times.get("checkout")),
        adults=adults,
        children=_int(data.get("children")),
        rooms=rooms,
        holder_name=" ".join(filter(None, (_text(holder.get("firstName")),
                                           _text(holder.get("lastName"))))) or None,
        holder_email=_text(holder.get("email")),
        holder_phone=_text(holder.get("phone")),
        total=float(price) if isinstance(price, (int, float)) else None,
        currency=currency,
        payment_status=_text(data.get("paymentStatus")),
        hotel_confirmation_code=_text(data.get("hotelConfirmationCode")),
        refundable=_refundable(tag),
        last_free_cancellation=_text(data.get("lastFreeCancellationDate")),
        cancellation_rules=_cancellation_rules(policies, currency),
        remarks=list(dict.fromkeys(remarks)),          # de-duplicate, keep order
        created_at=_text(data.get("createdAt")),
    )


def get_booking_details(booking_id: str, client: NuiteeClient | None = None) -> BookingDetails:
    """Retrieve a booking via GET /bookings/{bookingId} and parse it."""
    return parse_booking_details((client or NuiteeClient()).get_booking(booking_id))
