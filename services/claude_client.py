import json
from typing import Any, Optional

import anthropic

from config.settings import MAX_TOKENS, MODEL_ID


class ClaudeClient:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: Optional[list[dict]] = None,
        max_tokens: int = MAX_TOKENS,
    ) -> anthropic.types.Message:
        kwargs: dict[str, Any] = {
            "model": MODEL_ID,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self._client.messages.create(**kwargs)

    def structured_output(
        self,
        messages: list[dict],
        system: str,
        schema: dict,
        schema_name: str,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Call Claude and return the raw JSON string conforming to schema."""
        resp = self._client.messages.create(
            model=MODEL_ID,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        # Extract text content from response
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return "{}"

    def extract_text(self, response: anthropic.types.Message) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def extract_tool_call(
        self, response: anthropic.types.Message, tool_name: str
    ) -> Optional[dict]:
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        return None
