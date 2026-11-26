"""Provider-agnostic LLM wrapper.

Speaks the OpenAI chat-completions protocol, which Gemini, OpenAI, Groq and Ollama
all expose. Swap providers by changing LLM_BASE_URL / LLM_MODEL in .env.

stream() yields:
    ("text", "<delta>")                         as tokens arrive
    ("done", "<full text>", [tool_call, ...])   once, at the end
where tool_call = {"id": str, "name": str, "arguments": str(JSON)}
"""
import logging
from typing import AsyncIterator, Optional

from . import config

log = logging.getLogger("millie.llm")


class LLM:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not config.LLM_API_KEY:
                raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        return self._client

    async def stream(self, messages: list[dict], tools: Optional[list[dict]] = None) -> AsyncIterator[tuple]:
        client = self._get_client()
        kwargs = {"model": config.LLM_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        if not config.LLM_STREAM:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            text = msg.content or ""
            calls = [{"id": tc.id or f"call_{i}", "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                     for i, tc in enumerate(msg.tool_calls or [])]
            if text:
                yield ("text", text)
            yield ("done", text, calls)
            return

        resp = await client.chat.completions.create(stream=True, **kwargs)
        parts: list[str] = []
        calls: dict[int, dict] = {}
        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                parts.append(delta.content)
                yield ("text", delta.content)
            for tc in delta.tool_calls or []:
                idx = tc.index if tc.index is not None else (len(calls) - 1 if (not tc.id and calls) else len(calls))
                # Some providers reuse index 0 for parallel calls: a new id means a new call.
                if tc.id and idx in calls and calls[idx]["id"] and calls[idx]["id"] != tc.id:
                    idx = max(calls) + 1
                entry = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function:
                    if tc.function.name and not entry["name"]:
                        entry["name"] = tc.function.name
                    if tc.function.arguments:
                        entry["arguments"] += tc.function.arguments
        final = []
        for i, idx in enumerate(sorted(calls)):
            c = calls[idx]
            final.append({"id": c["id"] or f"call_{i}", "name": c["name"], "arguments": c["arguments"] or "{}"})
        yield ("done", "".join(parts), final)

    async def run(self, messages: list[dict], tools: Optional[list[dict]] = None) -> tuple[str, list[dict]]:
        """Consume the stream and return (text, tool_calls). Used by non-voice callers."""
        text, calls = "", []
        async for ev in self.stream(messages, tools):
            if ev[0] == "done":
                text, calls = ev[1], ev[2]
        return text, calls


def assistant_tool_message(text: str, calls: list[dict]) -> dict:
    """Build the assistant message that records which tools were called."""
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [{"id": c["id"], "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls],
    }


def tool_result_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}
