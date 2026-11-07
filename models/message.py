"""`messages` collection model."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal

from bson import ObjectId
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)
from pymongo import ASCENDING, IndexModel


def _to_object_id(value: Any) -> ObjectId:
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    raise ValueError(f"Invalid ObjectId: {value!r}")


# ObjectId in Python / MongoDB, string in JSON.
PyObjectId = Annotated[
    ObjectId,
    BeforeValidator(_to_object_id),
    PlainSerializer(str, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string", "pattern": "^[0-9a-f]{24}$"}),
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="forbid", str_strip_whitespace=True)

    collection_name: ClassVar[str] = "messages"
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)],
                   name="conversation_id_created_at"),
    ]

    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    conversation_id: PyObjectId                  # -> conversations._id
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=_utcnow)

    def to_document(self) -> dict[str, Any]:
        """Dict ready for MongoDB: `_id` key, native ObjectId and datetime values."""
        return self.model_dump(by_alias=True)
