"""`hotel_selections` collection: the hotel a user picked in a conversation.

Selecting a hotel is NOT a booking: no booking record and no PDF are created here.
One document per (conversation, hotel); picking the same hotel again updates it.
The latest `updated_at` in a conversation is its current selection. When a room is
held (prebook) its rate details are added and status becomes "room_held"; a
confirmed booking marks it "booked".
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from models.conversation import PyObjectId, _utcnow


class SelectedRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offer_id: Optional[str] = None
    prebook_id: Optional[str] = None
    room_name: Optional[str] = None
    board: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    refundable: Optional[bool] = None
    rooms: Optional[int] = None


class HotelSelection(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="forbid", str_strip_whitespace=True)

    collection_name: ClassVar[str] = "hotel_selections"
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("conversation_id", ASCENDING), ("hotel_id", ASCENDING)], unique=True,
                   name="conversation_hotel_unique"),
        IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)],
                   name="user_id_updated_at"),
    ]

    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId                      # -> users._id
    conversation_id: PyObjectId              # -> conversations._id (the trip)
    thread_id: str                           # -> conversations.thread_id
    hotel_id: str                            # Nuitee hotel id
    hotel_name: str
    address: Optional[str] = None
    city: Optional[str] = None
    stars: Optional[float] = None
    guest_rating: Optional[float] = None
    distance_km: Optional[float] = None      # from the preferred area
    rank: Optional[int] = None
    match_score: Optional[int] = None
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    adults: Optional[int] = None
    rooms: Optional[int] = None
    currency: Optional[str] = None
    room: Optional[SelectedRoom] = None      # set when a room is held (prebook)
    status: Literal["selected", "room_held", "booked"] = "selected"
    selected_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)
