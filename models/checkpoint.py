"""LangGraph checkpoint documents in MongoDB (read-only description).

The `checkpoints` and `checkpoint_writes` collections are created, indexed and
written by `MongoDBSaver` (langgraph-checkpoint-mongodb), configured in
services/checkpointing.py. These models document their shape and let you read
them (e.g. for inspection or a report); the application never writes them
directly, and they are NOT part of services.database.CHAT_MODELS, so
`ensure_indexes()` leaves them alone.

Relation to the chat collections:
    conversations.thread_id  ==  checkpoints.thread_id  ==  checkpoint_writes.thread_id
    checkpoint_ns = "workflow:<name>" (workflow:input, workflow:hotel, ...) keeps
    each workflow's history separate inside one conversation's thread.

Example:
    doc = db[CheckpointDoc.collection_name].find_one({"thread_id": thread_id})
    cp = CheckpointDoc.model_validate(doc)
    cp.workflow                        # "input" / "hotel"
    cp.state()["channel_values"]       # the graph state at that step
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from config.settings import CHECKPOINT_COLLECTION, CHECKPOINT_WRITES_COLLECTION

NS_PREFIX = "workflow:"     # same prefix as services.checkpointing.NS_PREFIX


def _serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return JsonPlusSerializer()


def _workflow(checkpoint_ns: str) -> Optional[str]:
    """"workflow:hotel" -> "hotel"; subgraph namespaces "workflow:hotel|x:1" -> "hotel"."""
    if not checkpoint_ns.startswith(NS_PREFIX):
        return None
    return checkpoint_ns[len(NS_PREFIX):].split("|", 1)[0]


class CheckpointDoc(BaseModel):
    """One saved step of a workflow run (collection `checkpoints`)."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="allow", frozen=True)

    collection_name: ClassVar[str] = CHECKPOINT_COLLECTION
    # Created by MongoDBSaver (listed here for reference only).
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("thread_id", ASCENDING), ("checkpoint_ns", ASCENDING),
                    ("checkpoint_id", DESCENDING)], unique=True),
    ]

    id: ObjectId = Field(alias="_id")
    thread_id: str                         # -> conversations.thread_id
    checkpoint_ns: str                     # "workflow:<name>"
    checkpoint_id: str                     # this step (time-ordered id)
    parent_checkpoint_id: Optional[str] = None   # previous step; None for the first
    type: str                              # serializer of `checkpoint`, e.g. "msgpack"
    checkpoint: bytes                      # whole graph state, serialized
    metadata: dict[str, Any] = Field(default_factory=dict)  # {source, step, parents, ...}

    @property
    def workflow(self) -> Optional[str]:
        return _workflow(self.checkpoint_ns)

    def state(self) -> dict[str, Any]:
        """Decoded checkpoint: {"channel_values": {...state...}, "versions_seen", ...}."""
        return _serde().loads_typed((self.type, self.checkpoint))


class CheckpointWriteDoc(BaseModel):
    """One value written by a node during a step (collection `checkpoint_writes`).

    Pending writes are how an interrupted step (a question waiting for the user)
    is resumed without re-running work that already finished.
    """

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True,
                              extra="allow", frozen=True)

    collection_name: ClassVar[str] = CHECKPOINT_WRITES_COLLECTION
    indexes: ClassVar[list[IndexModel]] = [
        IndexModel([("thread_id", ASCENDING), ("checkpoint_ns", ASCENDING),
                    ("checkpoint_id", DESCENDING), ("task_id", ASCENDING),
                    ("idx", ASCENDING)], unique=True),
    ]

    id: ObjectId = Field(alias="_id")
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str                     # -> checkpoints.checkpoint_id
    task_id: str                           # the node run that wrote it
    task_path: str = ""
    idx: int                               # order within the task (negative = special)
    channel: str                           # state key, e.g. "trip_extraction", "__interrupt__"
    type: str
    value: bytes                           # serialized value

    @property
    def workflow(self) -> Optional[str]:
        return _workflow(self.checkpoint_ns)

    def decoded_value(self) -> Any:
        return _serde().loads_typed((self.type, self.value))
