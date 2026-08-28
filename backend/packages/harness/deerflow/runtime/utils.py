"""Compatibility timestamp helpers for migrated Anaxa modules.

The DeerFlow runtime keeps its canonical implementations in
``deerflow.utils.time``.  Anaxa's academic and research modules historically
imported :func:`now_iso` from ``<package>.runtime.utils``; this small shim keeps
that import path stable while exposing the other shared time helpers as well.
"""

from deerflow.utils.time import coerce_iso, is_lease_expired, now_iso

__all__ = ["coerce_iso", "is_lease_expired", "now_iso"]
