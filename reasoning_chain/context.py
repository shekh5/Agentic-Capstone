"""Conversation context selection and Redis-backed rolling memory."""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from typing import Callable, Literal, Optional

try:
    from redis.exceptions import WatchError
except ImportError:  # pragma: no cover - Redis is an application dependency
    WatchError = RuntimeError

Role = Literal["user", "model"]
TokenCounter = Callable[[list[dict], str], int]
Summarizer = Callable[[str, list["ContextMessage"]], str]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ContextSettings:
    input_token_budget: int = field(
        default_factory=lambda: _env_int("CONTEXT_INPUT_TOKEN_BUDGET", 24_000)
    )
    output_token_reserve: int = field(
        default_factory=lambda: _env_int("CONTEXT_OUTPUT_TOKEN_RESERVE", 1_000)
    )
    token_safety_margin: int = field(
        default_factory=lambda: _env_int("CONTEXT_TOKEN_SAFETY_MARGIN", 1_000)
    )
    recent_messages: int = field(
        default_factory=lambda: _env_int("CONTEXT_RECENT_MESSAGES", 16)
    )
    recent_high_watermark: int = field(
        default_factory=lambda: _env_int("CONTEXT_RECENT_HIGH_WATERMARK", 20)
    )
    session_max_messages: int = field(
        default_factory=lambda: _env_int("SESSION_MAX_MESSAGES", 200)
    )
    session_ttl_seconds: int = field(
        default_factory=lambda: _env_int("SESSION_TTL_SECONDS", 2_592_000)
    )

    @property
    def usable_input_tokens(self) -> int:
        reserved = self.output_token_reserve + self.token_safety_margin
        return max(1, self.input_token_budget - reserved)

    @property
    def compaction_threshold(self) -> int:
        return max(self.recent_high_watermark, self.recent_messages)


@dataclass(frozen=True)
class ContextMessage:
    role: Role
    text: str
    timestamp: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"role": self.role, "text": self.text, "timestamp": self.timestamp},
            separators=(",", ":"),
        )

    @classmethod
    def from_raw(cls, raw: str | dict) -> Optional["ContextMessage"]:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            sender = str(data.get("role", data.get("sender", ""))).lower()
            role: Role = "model" if sender in {"agent", "assistant", "model"} else "user"
            if sender not in {"agent", "assistant", "model", "user"}:
                return None
            text = str(data.get("text", "")).strip()
            if not text:
                return None
            return cls(role=role, text=text, timestamp=str(data.get("timestamp", "")))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class ContextBundle:
    summary: str = ""
    recent: list[ContextMessage] = field(default_factory=list)


def estimate_tokens(contents: list[dict], system_instruction: str) -> int:
    """Conservative local fallback when Gemini token counting is unavailable."""
    serialized = system_instruction + json.dumps(contents, ensure_ascii=False)
    return max(1, math.ceil(len(serialized) / 3.5))


def _content(role: Role, text: str) -> dict:
    return {"role": role, "parts": [{"text": text}]}


def build_budgeted_contents(
    bundle: Optional[ContextBundle],
    current_prompt: str,
    system_instruction: str,
    settings: Optional[ContextSettings] = None,
    token_counter: Optional[TokenCounter] = None,
) -> list[dict]:
    """Build role-correct Gemini contents while keeping the current request mandatory."""
    settings = settings or ContextSettings()
    counter = token_counter or estimate_tokens
    current = _content("user", current_prompt)
    if bundle is None:
        return [current]

    selected: list[ContextMessage] = []
    candidates = bundle.recent[-settings.recent_messages :]
    for message in reversed(candidates):
        trial_messages = list(reversed([message, *selected]))
        trial = [_content(item.role, item.text) for item in trial_messages] + [current]
        if estimate_tokens(trial, system_instruction) <= settings.usable_input_tokens:
            selected.insert(0, message)
        else:
            break

    summary_content = None
    if bundle.summary:
        summary_text = (
            '<conversation_summary trust="untrusted">\n'
            f"{escape(bundle.summary, quote=False)}\n"
            "</conversation_summary>"
        )
        trial = [_content("user", summary_text)]
        trial.extend(_content(item.role, item.text) for item in selected)
        trial.append(current)
        if estimate_tokens(trial, system_instruction) <= settings.usable_input_tokens:
            summary_content = _content("user", summary_text)

    contents = [_content(item.role, item.text) for item in selected]
    if summary_content:
        contents.insert(0, summary_content)
    contents.append(current)

    try:
        while (
            len(contents) > 1
            and counter(contents, system_instruction) > settings.usable_input_tokens
        ):
            if summary_content and contents[0] is summary_content:
                contents.pop(0)
                summary_content = None
            else:
                contents.pop(0)
    except Exception:
        # The local estimate already enforced a conservative limit.
        pass
    return contents


class RedisContextStore:
    """Stores clean model context independently from rich UI/trace records."""

    def __init__(self, client, settings: Optional[ContextSettings] = None):
        self.client = client
        self.settings = settings or ContextSettings()

    @staticmethod
    def _keys(session_id: str) -> dict[str, str]:
        prefix = f"session:{session_id}"
        return {
            "display": f"{prefix}:messages",
            "recent": f"{prefix}:context:recent",
            "summary": f"{prefix}:context:summary",
            "meta": f"{prefix}:context:meta",
            "metadata": f"{prefix}:metadata",
        }

    def load(self, session_id: str) -> ContextBundle:
        keys = self._keys(session_id)
        recent_exists = self.client.exists(keys["recent"])
        meta_exists = self.client.exists(keys["meta"])
        if self._exists(recent_exists) or self._exists(meta_exists):
            raw_messages = self.client.lrange(keys["recent"], 0, -1)
        else:
            raw_messages = self._migrate_legacy_display_history(keys)
        messages = [ContextMessage.from_raw(raw) for raw in raw_messages]
        summary = self.client.get(keys["summary"]) or ""
        return ContextBundle(summary=str(summary), recent=[msg for msg in messages if msg])

    @staticmethod
    def _exists(value) -> bool:
        return value is True or isinstance(value, int) and value > 0

    def _migrate_legacy_display_history(self, keys: dict[str, str]) -> list[str]:
        legacy = self.client.lrange(keys["display"], 0, -1)
        clean = [message for raw in legacy if (message := ContextMessage.from_raw(raw))]
        pipe = self.client.pipeline()
        if clean:
            pipe.rpush(keys["recent"], *(message.to_json() for message in clean))
        pipe.set(
            keys["meta"],
            json.dumps({"version": 1, "migrated": True}),
            ex=self.settings.session_ttl_seconds,
        )
        pipe.expire(keys["recent"], self.settings.session_ttl_seconds)
        pipe.execute()
        return [message.to_json() for message in clean]

    def append(self, session_id: str, message: ContextMessage) -> None:
        keys = self._keys(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(keys["recent"], message.to_json())
        pipe.expire(keys["recent"], self.settings.session_ttl_seconds)
        pipe.expire(keys["summary"], self.settings.session_ttl_seconds)
        pipe.execute()

    def append_display(self, session_id: str, *records: dict) -> None:
        keys = self._keys(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(keys["display"], *(json.dumps(record) for record in records))
        pipe.ltrim(keys["display"], -self.settings.session_max_messages, -1)
        pipe.expire(keys["display"], self.settings.session_ttl_seconds)
        pipe.execute()

    def compact(self, session_id: str, summarizer: Summarizer) -> bool:
        """Summarize old messages. Leave Redis unchanged if generation fails."""
        keys = self._keys(session_id)
        snapshot = self.client.lrange(keys["recent"], 0, -1)
        if len(snapshot) <= self.settings.compaction_threshold:
            return False
        existing_summary = self.client.get(keys["summary"]) or ""
        keep_count = min(self.settings.recent_messages, len(snapshot))
        old_raw, retained = snapshot[:-keep_count], snapshot[-keep_count:]
        old_messages = [ContextMessage.from_raw(raw) for raw in old_raw]
        old_messages = [message for message in old_messages if message]
        if not old_messages:
            return False
        new_summary = summarizer(str(existing_summary), old_messages).strip()
        if not new_summary:
            return False

        # Optimistic concurrency: compact only if no request appended during summarization.
        pipe = self.client.pipeline()
        try:
            pipe.watch(keys["recent"], keys["summary"])
            current_summary = pipe.get(keys["summary"]) or ""
            if pipe.lrange(keys["recent"], 0, -1) != snapshot:
                pipe.unwatch()
                return False
            if str(current_summary) != str(existing_summary):
                pipe.unwatch()
                return False
            pipe.multi()
            pipe.delete(keys["recent"])
            if retained:
                pipe.rpush(keys["recent"], *retained)
            pipe.set(keys["summary"], new_summary, ex=self.settings.session_ttl_seconds)
            pipe.set(
                keys["meta"],
                json.dumps(
                    {
                        "version": 1,
                        "summarized_at": datetime.now(timezone.utc).isoformat(),
                        "summarized_messages": len(old_messages),
                    }
                ),
                ex=self.settings.session_ttl_seconds,
            )
            pipe.expire(keys["recent"], self.settings.session_ttl_seconds)
            pipe.execute()
            return True
        except WatchError:
            return False
        finally:
            reset = getattr(pipe, "reset", None)
            if reset:
                reset()
