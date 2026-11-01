"""
Gemini LLM client.

Usage:
    initialize_client(api_key)   # call once at startup
    client = get_client()        # call in any node
    resp = client.chat(messages, system, tools)
    resp.text          # assistant text
    resp.tool_calls    # [{"name": str, "input": dict}]
"""
from __future__ import annotations

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
    def structured_output(
        self, messages: list[dict], system: str = "", schema: Optional[dict] = None
    ) -> str: ...


# ── Gemini (google-genai SDK) ─────────────────────────────────────────────────

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model_id: str):
        from google import genai
        from google.genai import types
        # Without a timeout an overloaded model hangs the request forever.
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=30_000)
        )
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
        """Convert message history to Gemini Contents, enforcing strict role alternation.

        Gemini requires: roles must alternate user/model. Consecutive same-role
        messages are merged into one Content with multiple text parts.
        Empty content is skipped to avoid API rejection.
        """
        from google.genai import types

        normalised: list[tuple[str, str]] = []  # (gemini_role, text)
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            content = m.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("text")
                )
            text = str(content).strip()
            if not text:
                continue  # skip empty messages — Gemini rejects empty parts
            if normalised and normalised[-1][0] == role:
                # Merge consecutive same-role messages
                normalised[-1] = (role, normalised[-1][1] + "\n" + text)
            else:
                normalised.append((role, text))

        return [
            types.Content(role=role, parts=[types.Part(text=text)])
            for role, text in normalised
        ]

    def _extract_response(self, response) -> tuple[str, list[dict]]:
        """Pull text and tool_calls out of a Gemini GenerateContentResponse."""
        text = ""
        tool_calls: list[dict] = []

        # Primary path: convenience accessors
        try:
            for fn_call in (response.function_calls or []):
                args = dict(fn_call.args) if fn_call.args else {}
                tool_calls.append({"name": fn_call.name, "input": args})
        except Exception:
            pass

        try:
            if response.text:
                text = response.text
        except Exception:
            pass

        # Fallback: iterate parts manually
        if not text and not tool_calls and response.candidates:
            candidate = response.candidates[0]
            finish = getattr(candidate, "finish_reason", None)
            # finish_reason 2 = MAX_TOKENS, 3 = SAFETY — surface these clearly
            if finish and str(finish) not in ("FinishReason.STOP", "1", "STOP"):
                raise RuntimeError(
                    f"Gemini response stopped unexpectedly: finish_reason={finish}. "
                    "Check safety settings or reduce prompt length."
                )
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    text = part.text
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    tool_calls.append({
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    })

        return text, tool_calls

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        from google.genai import types

        contents = self._build_contents(messages)
        cfg_kwargs: dict = {}
        if system:
            cfg_kwargs["system_instruction"] = system
        if tools:
            cfg_kwargs["tools"] = self._build_tools(tools)

        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
            )
        except Exception as exc:
            _raise_gemini_error(exc)

        text, tool_calls = self._extract_response(response)
        return LLMResponse(text=text, tool_calls=tool_calls)

    def structured_output(self, messages, system="", schema: Optional[dict] = None) -> str:
        from google.genai import types

        contents = self._build_contents(messages)
        cfg_kwargs: dict = {
            "system_instruction": (system or "") + "\n\nReturn ONLY valid JSON. No markdown, no prose.",
            "response_mime_type": "application/json",
        }
        if schema is not None:
            try:
                cfg_kwargs["response_schema"] = self._build_schema(schema)
            except Exception:
                pass  # schema build failure is non-fatal — fall back to prompt-only JSON mode
        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        except Exception as exc:
            _raise_gemini_error(exc)

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


def _raise_gemini_error(exc: Exception) -> None:
    """Re-raise Gemini API errors with actionable messages."""
    msg = str(exc)
    if "API_KEY_INVALID" in msg or "PermissionDenied" in msg or "403" in msg:
        raise RuntimeError(
            "Gemini API key is invalid or missing. "
            "Set GEMINI_API_KEY in your .env file."
        ) from exc
    if "ResourceExhausted" in msg or "429" in msg or "quota" in msg.lower():
        raise RuntimeError(
            "Gemini rate limit or quota exceeded. "
            "Wait a moment or check https://aistudio.google.com for quota details."
        ) from exc
    if "NOT_FOUND" in msg or "404" in msg:
        raise RuntimeError(
            f"Gemini model not found. Check GEMINI_MODEL in your .env — "
            "recommended model: gemini-3.5-flash-lite"
        ) from exc
    raise RuntimeError(f"Gemini API error: {exc}") from exc





# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[LLMClient] = None


def initialize_client(api_key: str = "") -> None:
    global _instance
    import os
    from config.settings import GEMINI_MODEL_ID
    model_id = os.getenv("GEMINI_MODEL", GEMINI_MODEL_ID)
    _instance = GeminiClient(api_key=api_key, model_id=model_id)


def get_client() -> LLMClient:
    if _instance is None:
        raise RuntimeError("LLM client not initialized. Call initialize_client() first.")
    return _instance
