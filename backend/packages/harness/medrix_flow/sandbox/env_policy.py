"""Environment policy for local sandbox subprocesses.

Sandbox commands should not inherit platform credentials from the Gateway
process. Keep ordinary runtime variables, remove credential-like names, and
allow explicit callers to add narrowly scoped values when needed.
"""

from __future__ import annotations

import fnmatch
import os

_SECRET_NAME_PATTERNS: tuple[str, ...] = (
    "*KEY*",
    "*SECRET*",
    "*TOKEN*",
    "*PASS*",
    "*CREDENTIAL*",
)
_BLOCKED_EXACT_NAMES = frozenset(
    {
        "DATABASE_URL",
        "DATABASE_URI",
        "REDIS_URL",
        "MONGODB_URI",
        "MONGO_URL",
        "AMQP_URL",
        "RABBITMQ_URL",
        "MYSQL_PWD",
        "REDISCLI_AUTH",
        "PGSERVICEFILE",
    }
)


def is_blocked_env_name(name: str) -> bool:
    """Return whether an environment variable looks credential-bearing."""
    upper = name.upper()
    return upper in _BLOCKED_EXACT_NAMES or any(
        fnmatch.fnmatchcase(upper, pattern) for pattern in _SECRET_NAME_PATTERNS
    )


def build_sandbox_env(injected: dict[str, str] | None = None) -> dict[str, str]:
    """Build a scrubbed subprocess environment, layering authorized values."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not is_blocked_env_name(name)
    }
    if injected:
        env.update(injected)
    return env
