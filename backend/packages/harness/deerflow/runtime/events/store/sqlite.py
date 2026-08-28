"""Compatibility SQLite event store for Anaxa research finalization.

DeerFlow's primary event store is SQLAlchemy-backed and lives in
``deerflow.runtime.events.store.db``.  The research assistant also needs a
small, self-contained store for its optional finalization sideband records;
keep that legacy schema available under the DeerFlow namespace without
coupling the core store to the old API.
"""

from medrix_flow.runtime.events.store.sqlite import SQLiteRunEventStore

__all__ = ["SQLiteRunEventStore"]
