"""Middleware for automatic thread title generation."""

import logging
import re
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from medrix_flow.config.title_config import get_title_config
from medrix_flow.models import create_chat_model

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_TITLE_LABEL_RE = re.compile(r"^(?:title|标题)\s*[:：-]\s*", re.IGNORECASE)
_TITLE_INTRO_RE = re.compile(
    r"^(?:here(?:'s| is)\s+(?:the\s+)?title|the\s+title\s+is)\s*[:：-]\s*",
    re.IGNORECASE,
)
_PROMPT_ECHO_RE = re.compile(
    r"generate a concise title|return only the title|user message:|assistant summary:|^the user\b|^the assistant\b",
    re.IGNORECASE,
)
_GENERIC_SUMMARY_TITLE_RE = re.compile(
    r"^(?:"
    r"here(?:'s| is)\s+(?:a\s+)?(?:brief\s+|concise\s+)?summary\s+of\s+(?:the\s+)?(?:conversation|chat)(?:\s+to\s+date)?"
    r"|(?:the\s+)?(?:conversation|chat)\s+summary"
    r"|summary\s+of\s+(?:this|the)\s+(?:conversation|chat)"
    r"|(?:本次|这次|当前)?(?:对话|聊天)(?:的)?(?:总结|小结)"
    r")\s*[:：。.!！-]*$",
    re.IGNORECASE,
)


class TitleMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    title: NotRequired[str | None]


class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """Automatically generate a title for the thread after the first user message."""

    state_schema = TitleMiddlewareState

    def _normalize_content(self, content: object) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [self._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

            nested_content = content.get("content")
            if nested_content is not None:
                return self._normalize_content(nested_content)

        return ""

    def _should_generate_title(self, state: TitleMiddlewareState) -> bool:
        """Check if we should generate a title for this thread."""
        config = get_title_config()
        if not config.enabled:
            return False

        # Keep clean user-edited titles, but allow bad model traces/prompt echoes to be replaced.
        if self._parse_title(state.get("title")):
            return False

        # Check if this is the first turn (has at least one user message and one assistant response)
        messages = state.get("messages", [])
        if len(messages) < 2:
            return False

        # Count user and assistant messages
        user_messages = [m for m in messages if m.type == "human"]
        assistant_messages = [m for m in messages if m.type == "ai"]

        # Generate a short conversation summary after an assistant response when no
        # usable title exists. This also repairs legacy titles polluted by traces.
        return len(user_messages) >= 1 and len(assistant_messages) >= 1

    def _build_title_prompt(self, state: TitleMiddlewareState) -> tuple[str, str]:
        """Extract user/assistant messages and build the title prompt.

        Returns (prompt_string, user_msg) so callers can use user_msg as fallback.
        """
        config = get_title_config()
        messages = state.get("messages", [])

        user_msg_content = next((m.content for m in messages if m.type == "human"), "")
        assistant_msg_content = next((m.content for m in messages if m.type == "ai"), "")

        user_msg = self._normalize_content(user_msg_content)
        assistant_msg = self._normalize_content(assistant_msg_content)

        prompt = config.prompt_template.format(
            max_words=config.max_words,
            max_chars=config.max_chars,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        return prompt, user_msg

    def _parse_title(self, content: object) -> str:
        """Normalize model output into a clean title string."""
        config = get_title_config()
        title_content = _THINK_BLOCK_RE.sub("\n", self._normalize_content(content))
        if _THINK_OPEN_RE.search(title_content):
            return ""

        title = ""
        for line in title_content.splitlines():
            candidate = line.strip().strip('"').strip("'").strip("`*_ ")
            candidate = _TITLE_INTRO_RE.sub("", candidate)
            candidate = _TITLE_LABEL_RE.sub("", candidate).strip().strip('"').strip("'")
            if not candidate or _PROMPT_ECHO_RE.search(candidate) or _GENERIC_SUMMARY_TITLE_RE.search(candidate):
                continue
            title = re.sub(r"\s+", " ", candidate).strip()
            break

        return title[: config.max_chars] if len(title) > config.max_chars else title

    def _fallback_title(self, user_msg: str) -> str:
        config = get_title_config()
        fallback_chars = min(config.max_chars, 50)
        if len(user_msg) > fallback_chars:
            return user_msg[:fallback_chars].rstrip() + "..."
        return user_msg if user_msg else "New Conversation"

    def _generate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """Synchronously generate a title. Returns state update or None."""
        existing_title = state.get("title")
        cleaned_existing_title = self._parse_title(existing_title)
        if existing_title and cleaned_existing_title:
            if cleaned_existing_title != existing_title:
                return {"title": cleaned_existing_title}
            return None

        if not self._should_generate_title(state):
            return None

        prompt, user_msg = self._build_title_prompt(state)
        config = get_title_config()
        model = create_chat_model(name=config.model_name, thinking_enabled=False)

        try:
            response = model.invoke(prompt)
            title = self._parse_title(response.content)
            if not title:
                title = self._fallback_title(user_msg)
        except Exception:
            logger.exception("Failed to generate title (sync)")
            title = self._fallback_title(user_msg)

        return {"title": title}

    async def _agenerate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """Asynchronously generate a title. Returns state update or None."""
        existing_title = state.get("title")
        cleaned_existing_title = self._parse_title(existing_title)
        if existing_title and cleaned_existing_title:
            if cleaned_existing_title != existing_title:
                return {"title": cleaned_existing_title}
            return None

        if not self._should_generate_title(state):
            return None

        prompt, user_msg = self._build_title_prompt(state)
        config = get_title_config()
        model = create_chat_model(name=config.model_name, thinking_enabled=False)

        try:
            response = await model.ainvoke(prompt)
            title = self._parse_title(response.content)
            if not title:
                title = self._fallback_title(user_msg)
        except Exception:
            logger.exception("Failed to generate title (async)")
            title = self._fallback_title(user_msg)

        return {"title": title}

    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return self._generate_title_result(state)

    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return await self._agenerate_title_result(state)
