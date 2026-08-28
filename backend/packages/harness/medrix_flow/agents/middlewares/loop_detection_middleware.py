"""Middleware to detect and break repetitive tool call loops.

P0 safety: prevents the agent from calling the same tool with the same
arguments indefinitely until the recursion limit kills the run, while still
letting unfinished tasks recover through a different route.

Detection strategy:
  1. After each model response, hash the tool calls (name + args).
  2. Track recent hashes in a sliding window.
  3. If the same hash appears >= warn_threshold times, inject a recovery
     warning that tells the model to stop repeating that exact call and choose
     a different route.
  4. If it appears >= hard_limit times, inject a stronger recovery directive
     that blocks the repeated call path and asks the model to inspect existing
     artifacts/errors/todos before continuing with alternate parameters, a
     different tool, or the next unfinished task.
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict, defaultdict
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Defaults — can be overridden via constructor
_DEFAULT_WARN_THRESHOLD = 3  # inject warning after 3 identical calls
_DEFAULT_HARD_LIMIT = 5  # force-stop after 5 identical calls
_DEFAULT_WINDOW_SIZE = 20  # track last N tool calls
_DEFAULT_MAX_TRACKED_THREADS = 100  # LRU eviction limit


def _hash_tool_calls(tool_calls: list[dict]) -> str:
    """Deterministic hash of a set of tool calls (name + args).

    This is intended to be order-independent: the same multiset of tool calls
    should always produce the same hash, regardless of their input order.
    """
    # First normalize each tool call to a minimal (name, args) structure.
    normalized: list[dict] = []
    for tc in tool_calls:
        normalized.append(
            {
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
        )

    # Sort by both name and a deterministic serialization of args so that
    # permutations of the same multiset of calls yield the same ordering.
    normalized.sort(
        key=lambda tc: (
            tc["name"],
            json.dumps(tc["args"], sort_keys=True, default=str),
        )
    )
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


_WARNING_MSG = (
    "[LOOP DETECTED] You are repeating the same tool calls. "
    "Do not repeat the identical call. Inspect the existing results, errors, "
    "artifacts, and unfinished todos, then continue via a different route: "
    "change the parameters, use another tool, repair the failed input, or move "
    "to the next unfinished step."
)

_HARD_STOP_MSG = (
    "[LOOP RECOVERY REQUIRED] The identical tool call exceeded the safety limit. "
    "That repeated call path is blocked. Do not present this as completed work. "
    "Read the available artifacts/audits/errors/todos, choose an alternate repair "
    "path or the next unfinished step, and continue until the requested deliverables "
    "are generated and verified or a real external blocker requires user input."
)


class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    """Detects and breaks repetitive tool call loops.

    Args:
        warn_threshold: Number of identical tool call sets before injecting
            a warning message. Default: 3.
        hard_limit: Number of identical tool call sets before stripping
            tool_calls entirely. Default: 5.
        window_size: Size of the sliding window for tracking calls.
            Default: 20.
        max_tracked_threads: Maximum number of threads to track before
            evicting the least recently used. Default: 100.
    """

    def __init__(
        self,
        warn_threshold: int = _DEFAULT_WARN_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
    ):
        super().__init__()
        self.warn_threshold = warn_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads
        self._lock = threading.Lock()
        # Per-thread tracking using OrderedDict for LRU eviction
        self._history: OrderedDict[str, list[str]] = OrderedDict()
        self._warned: dict[str, set[str]] = defaultdict(set)

    def _get_thread_id(self, runtime: Runtime) -> str:
        """Extract thread_id from runtime context for per-thread tracking."""
        thread_id = runtime.context.get("thread_id")
        if thread_id:
            return thread_id
        return "default"

    def _evict_if_needed(self) -> None:
        """Evict least recently used threads if over the limit.

        Must be called while holding self._lock.
        """
        while len(self._history) > self.max_tracked_threads:
            evicted_id, _ = self._history.popitem(last=False)
            self._warned.pop(evicted_id, None)
            logger.debug("Evicted loop tracking for thread %s (LRU)", evicted_id)

    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """Track tool calls and check for loops.

        Returns:
            (warning_message_or_none, should_hard_stop)
        """
        messages = state.get("messages", [])
        if not messages:
            return None, False

        last_msg = messages[-1]
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None, False

        thread_id = self._get_thread_id(runtime)
        call_hash = _hash_tool_calls(tool_calls)

        with self._lock:
            # Touch / create entry (move to end for LRU)
            if thread_id in self._history:
                self._history.move_to_end(thread_id)
            else:
                self._history[thread_id] = []
                self._evict_if_needed()

            history = self._history[thread_id]
            history.append(call_hash)
            if len(history) > self.window_size:
                history[:] = history[-self.window_size :]

            count = history.count(call_hash)
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            if count >= self.hard_limit:
                logger.error(
                    "Loop hard limit reached — forcing stop",
                    extra={
                        "thread_id": thread_id,
                        "call_hash": call_hash,
                        "count": count,
                        "tools": tool_names,
                    },
                )
                return _HARD_STOP_MSG, True

            if count >= self.warn_threshold:
                warned = self._warned[thread_id]
                if call_hash not in warned:
                    warned.add(call_hash)
                    logger.warning(
                        "Repetitive tool calls detected — injecting warning",
                        extra={
                            "thread_id": thread_id,
                            "call_hash": call_hash,
                            "count": count,
                            "tools": tool_names,
                        },
                    )
                    return _WARNING_MSG, False
                # Warning already injected for this hash — suppress
                return None, False

        return None, False

    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        warning, hard_stop = self._track_and_check(state, runtime)

        if hard_stop:
            # Inject a recovery directive after the repeated AI tool call. This
            # blocks the repeated call path from proceeding while keeping the run
            # oriented toward repair/continuation instead of premature delivery.
            return {"messages": [SystemMessage(content=_HARD_STOP_MSG)]}

        if warning:
            # Inject a system message warning the model
            return {"messages": [SystemMessage(content=warning)]}

        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    def reset(self, thread_id: str | None = None) -> None:
        """Clear tracking state. If thread_id given, clear only that thread."""
        with self._lock:
            if thread_id:
                self._history.pop(thread_id, None)
                self._warned.pop(thread_id, None)
            else:
                self._history.clear()
                self._warned.clear()
