"""`users` collection model."""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Annotated, Any, ClassVar, Literal, Optional
from uuid import uuid4

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


# ── Learned profile (stored inside the user's document) ─────────────────────
# Filled only by the profile learner (services/user_profile.py) from what the user
# explicitly said in their chats. Lasting facts only, never trip details such as
# budget, number of travellers, dates or destination.

ProfileCategory = Literal["usual_departure_point", "food", "interest", "pace", "hotel",
                          "hotel_stars", "accessibility"]
# Personal needs count from the first explicit mention; habits need a second chat
# (or an explicit "I always ..." / "I live in ...").
HARD_CATEGORIES = {"food", "accessibility"}
SINGLE_VALUE_CATEGORIES = {"usual_departure_point", "pace", "hotel_stars"}
CONFIRM_AFTER_CHATS = 2


class ProfileFact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: ProfileCategory
    value: str = Field(min_length=1)
    explicit: bool = False                       # stated as lasting ("I live in Pune")
    quotes: list[str] = Field(default_factory=list)            # the user's own words
    conversation_ids: list[str] = Field(default_factory=list)  # chats it came from
    first_learned_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @property
    def confirmed(self) -> bool:
        return (self.category in HARD_CATEGORIES or self.explicit
                or len(self.conversation_ids) >= CONFIRM_AFTER_CHATS)


class Companion(BaseModel):
    """Someone the user travels with, e.g. Priya (wife), 28 as of 2026-11-08."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: Optional[str] = None
    relation: Optional[str] = None               # "wife", "mother", "friend", ...
    age: Optional[int] = Field(default=None, ge=0, le=120)   # as stated ...
    age_stated_on: Optional[str] = None      # ... on this day (YYYY-MM-DD; BSON has no date)
    notes: list[str] = Field(default_factory=list)   # their own needs: "vegan", "can't walk far"
    quotes: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def age_on(self, today: date) -> Optional[int]:
        """The stated age moved forward by the full years since it was stated."""
        if self.age is None or not self.age_stated_on:
            return self.age
        s = date.fromisoformat(self.age_stated_on)
        years = today.year - s.year - ((today.month, today.day) < (s.month, s.day))
        return self.age + max(0, years)

    def label(self, today: date) -> str:
        """"Priya (wife, 28)", "Mother (60)", "Rahul"."""
        age = self.age_on(today)
        if self.name:
            extra = [x for x in (self.relation, None if age is None else str(age)) if x]
            return f"{self.name} ({', '.join(extra)})" if extra else self.name
        head = (self.relation or "companion").capitalize()
        return head if age is None else f"{head} ({age})"

    def summary(self, today: date) -> str:
        """Label plus their needs, for the agents: "Mother (62): can't walk far"."""
        return self.label(today) + (f": {', '.join(self.notes)}" if self.notes else "")


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[ProfileFact] = Field(default_factory=list)
    learned_from: list[str] = Field(default_factory=list)   # conversations already read
    updated_at: Optional[datetime] = None


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
    profile: UserProfile = Field(default_factory=UserProfile)      # learned, never typed in
    companions: list[Companion] = Field(default_factory=list)      # people they travel with

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
