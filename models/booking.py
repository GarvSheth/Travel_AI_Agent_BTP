"""`bookings` collection: bookings the provider has CONFIRMED.

Only provider-confirmed bookings are stored (never a selection or a hold).
`booking_type` / `provider` keep the schema open for flights etc. later.
(provider, provider_booking_id) is unique, so a repeated confirmation / retry
updates the same record instead of creating a duplicate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from models.conversation import PyObjectId, _utcnow


class Booking(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="forbid", str_strip_whitespace=True)

    collection_name: ClassVar[str] = "bookings"
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("provider", ASCENDING), ("provider_booking_id", ASCENDING)], unique=True,
                   name="provider_booking_unique"),
        IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)],
                   name="user_id_created_at"),
        IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)],
                   name="conversation_id_created_at"),
    ]

    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId                      # -> users._id
    conversation_id: PyObjectId              # -> conversations._id
    thread_id: str                           # -> conversations.thread_id
    booking_type: Literal["hotel", "flight"] = "hotel"
    provider: str = "nuitee"
    provider_booking_id: str
    status: str                              # provider status, e.g. "CONFIRMED"
    confirmed: bool                          # provider-confirmed (always True when stored)
    sandbox: bool = False                    # test booking: nothing reserved or charged
    item_id: Optional[str] = None            # e.g. Nuitee hotel id
    title: Optional[str] = None              # e.g. hotel name
    start_date: Optional[str] = None         # check-in
    end_date: Optional[str] = None           # check-out
    total: Optional[float] = None
    currency: Optional[str] = None
    holder_name: Optional[str] = None
    holder_email: Optional[str] = None
    hotel_confirmation_code: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)   # full parsed provider record
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)
