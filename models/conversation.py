"""`conversations` collection model (one per LangGraph thread)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar

from bson import ObjectId
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)
from pymongo import ASCENDING, DESCENDING, IndexModel


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


class Conversation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="forbid", str_strip_whitespace=True)

    collection_name: ClassVar[str] = "conversations"
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("thread_id", ASCENDING)], unique=True, name="thread_id_unique"),
        IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)],
                   name="user_id_updated_at"),
    ]

    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    user_id: PyObjectId                          # -> users._id
    thread_id: str = Field(min_length=1)
    title: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def to_document(self) -> dict[str, Any]:
        """Dict ready for MongoDB: `_id` key, native ObjectId and datetime values."""
        return self.model_dump(by_alias=True)
