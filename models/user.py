"""`users` collection model."""
from __future__ import annotations

import re
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
    field_validator,
)
from pymongo import ASCENDING, IndexModel

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,32}$")


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


class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="forbid", str_strip_whitespace=True)

    collection_name: ClassVar[str] = "users"
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
        # sparse: users saved before login existed have no username
        IndexModel([("username", ASCENDING)], unique=True, sparse=True, name="username_unique"),
    ]

    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    username: str
    name: str = Field(min_length=1)
    email: str
    password_hash: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("username")
    @classmethod
    def _valid_username(cls, value: str) -> str:
        value = value.strip().lower()            # lowercase so login is case-insensitive
        if not _USERNAME_RE.match(value):
            raise ValueError("Username must be 3-32 characters: letters, digits, . _ -")
        return value

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        value = value.strip().lower()            # lowercase so the unique index is case-insensitive
        if not _EMAIL_RE.match(value):
            raise ValueError("Invalid email address")
        return value

    def to_document(self) -> dict[str, Any]:
        """Dict ready for MongoDB: `_id` key, native ObjectId and datetime values."""
        return self.model_dump(by_alias=True)
