"""
Unified LLM client. Supports Anthropic and Gemini behind the same interface.

Usage:
    initialize_client("gemini", api_key)   # call once at startup
    client = get_client()                  # call in any node
    resp = client.chat(messages, system, tools)
    resp.text          # assistant text
    resp.tool_calls    # [{"name": str, "input": dict}]
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional


class LLMResponse:
    def __init__(self, text: str, tool_calls: list[dict]):
        self.text = text
        self.tool_calls = tool_calls  # [{"name": str, "input": dict}]

    def get_tool_call(self, name: str) -> Optional[dict]:
        for tc in self.tool_calls:
            if tc["name"] == name:
                return tc["input"]
        return None


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def structured_output(self, messages: list[dict], system: str = "") -> str: ...


# ── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model_id: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = model_id

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)

        text = ""
        tool_calls = []
        for block in resp.content:
            if hasattr(block, "text"):
                text = block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append({"name": block.name, "input": block.input})
        return LLMResponse(text=text, tool_calls=tool_calls)

    def structured_output(self, messages, system="") -> str:
        resp = self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return "{}"


# ── Gemini (google-genai SDK) ─────────────────────────────────────────────────

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model_id: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id

    # ── Schema helpers ────────────────────────────────────────────────────────

    def _build_schema(self, schema_dict: dict):
        from google.genai import types
        TYPE_MAP = {
            "string": types.Type.STRING,   "STRING": types.Type.STRING,
            "integer": types.Type.INTEGER, "INTEGER": types.Type.INTEGER,
            "number": types.Type.NUMBER,   "NUMBER": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN, "BOOLEAN": types.Type.BOOLEAN,
            "array": types.Type.ARRAY,     "ARRAY": types.Type.ARRAY,
            "object": types.Type.OBJECT,   "OBJECT": types.Type.OBJECT,
        }
        kwargs: dict = {}
        if "type" in schema_dict:
            kwargs["type"] = TYPE_MAP.get(schema_dict["type"], types.Type.STRING)
        if "description" in schema_dict:
            kwargs["description"] = schema_dict["description"]
        if "enum" in schema_dict:
            kwargs["enum"] = schema_dict["enum"]
        if "required" in schema_dict:
            kwargs["required"] = schema_dict["required"]
        if "properties" in schema_dict:
            kwargs["properties"] = {
                k: self._build_schema(v) for k, v in schema_dict["properties"].items()
            }
        if "items" in schema_dict:
            kwargs["items"] = self._build_schema(schema_dict["items"])
        return types.Schema(**kwargs)

    def _build_tools(self, tools: list[dict]):
        from google.genai import types
        declarations = []
        for t in tools:
            declarations.append(types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=self._build_schema(t.get("input_schema", {})),
            ))
        return [types.Tool(function_declarations=declarations)]

    def _build_contents(self, messages: list[dict]):
        from google.genai import types
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("text")
                )
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=str(content))],
            ))
        return contents

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        from google.genai import types

        contents = self._build_contents(messages)
        cfg_kwargs: dict = {}
        if system:
            cfg_kwargs["system_instruction"] = system
        if tools:
            cfg_kwargs["tools"] = self._build_tools(tools)

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
        )

        text = ""
        tool_calls: list[dict] = []

        # Collect function calls via the convenience accessor first
        for fn_call in (response.function_calls or []):
            args = dict(fn_call.args) if fn_call.args else {}
            tool_calls.append({"name": fn_call.name, "input": args})

        # Extract text from candidates
        try:
            if response.text:
                text = response.text
        except Exception:
            pass

        # Fallback: iterate parts manually (catches function_call if accessor missing)
        if not text and not tool_calls and response.candidates:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    text = part.text
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    tool_calls.append({
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    })

        return LLMResponse(text=text, tool_calls=tool_calls)

    def structured_output(self, messages, system="") -> str:
        from google.genai import types

        contents = self._build_contents(messages)
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(system or "") + "\n\nReturn ONLY valid JSON. No markdown, no prose.",
                response_mime_type="application/json",
            ),
        )
        text = ""
        try:
            text = response.text or ""
        except Exception:
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "text", None):
                        text = part.text
                        break
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return text.strip()


# ── OpenRouter (OpenAI-compatible) ───────────────────────────────────────────

class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str, model_id: str):
        from openai import OpenAI
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._model_id = model_id

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Anthropic input_schema format → OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        kwargs: dict = {"model": self._model_id, "messages": msgs}
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text = message.content or ""
        tool_calls: list[dict] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append({"name": tc.function.name, "input": args})

        return LLMResponse(text=text, tool_calls=tool_calls)

    def structured_output(self, messages, system="") -> str:
        msgs = []
        full_system = (system or "") + "\n\nReturn ONLY valid JSON. No markdown, no prose."
        msgs.append({"role": "system", "content": full_system})
        msgs.extend(messages)

        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=msgs,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback for models that don't support response_format
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=msgs,
            )

        text = response.choices[0].message.content or "{}"
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return text.strip()


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[LLMClient] = None


def initialize_client(provider: str, api_key: str) -> None:
    global _instance
    from config.settings import ANTHROPIC_MODEL_ID, GEMINI_MODEL_ID, OPENROUTER_MODEL_ID
    if provider == "anthropic":
        _instance = AnthropicClient(api_key=api_key, model_id=ANTHROPIC_MODEL_ID)
    elif provider == "openrouter":
        _instance = OpenRouterClient(api_key=api_key, model_id=OPENROUTER_MODEL_ID)
    else:
        _instance = GeminiClient(api_key=api_key, model_id=GEMINI_MODEL_ID)


def get_client() -> LLMClient:
    if _instance is None:
        raise RuntimeError("LLM client not initialized. Call initialize_client() first.")
    return _instance
