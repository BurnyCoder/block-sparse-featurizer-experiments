"""Locked server-side experiment sessions with cancellation and TTL cleanup.

Global context
--------------
Models and activation arrays stay in Python memory rather than browser state.
Each session owns a reentrant state lock and a cooperative cancellation token;
the registry separately locks its ID map. Python recommends context-managed
``RLock`` usage and ``Event`` for graceful thread signalling:
https://docs.python.org/3/library/threading.html#rlock-objects and
https://docs.python.org/3/library/threading.html#event-objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import re
import threading
import time
from uuid import uuid4

from .types import ExperimentState


_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validated_session_id(value: str | None) -> str:
    """Return a bounded opaque identifier that is safe in logs and URL state."""

    session_id = uuid4().hex if value is None else value
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError(
            "session_id must be 1-128 letters, digits, dots, underscores, or hyphens"
        )
    return session_id


@dataclass(slots=True)
class ExperimentSession:
    """Mutable state and synchronization primitives for one browser session."""

    state: ExperimentState
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def session_id(self) -> str:
        """Return the stable opaque ID shared with the lightweight browser state."""

        return self.state.session_id

    @contextmanager
    def locked_state(self) -> Iterator[ExperimentState]:
        """Yield the state under its reentrant lock and refresh activity on exit."""

        with self._lock:
            try:
                yield self.state
            finally:
                self.state.touch()

    def cancellation_token(self) -> threading.Event:
        """Return the current operation token for a worker to retain and poll.

        A worker must retain this returned object for its whole operation. Reset
        replaces the session's token after setting the old one, so dynamically
        looking the token up again could observe the next operation's clear flag.
        """

        with self._lock:
            return self._cancel_event

    def request_cancel(self) -> None:
        """Cooperatively signal every worker holding the current operation token."""

        # ``Event.set`` is thread-safe and intentionally does not wait for the
        # state lock, allowing cancellation while a phase owns state mutation.
        self._cancel_event.set()

    def reset(self) -> None:
        """Cancel current work, release large state, and install a fresh token."""

        # Signal before waiting for the state lock so a worker that owns the
        # lock can observe cancellation and finish its current phase promptly.
        self.request_cancel()
        with self._lock:
            # A concurrent reset may have replaced the token while this caller
            # waited for the lock; signal that token as well before replacing it.
            self._cancel_event.set()
            session_id = self.state.session_id
            self.state = ExperimentState(session_id=session_id)
            self._cancel_event = threading.Event()


@dataclass(slots=True)
class _RegistryEntry:
    """Pair a session with monotonic last-access time used only by the registry."""

    session: ExperimentSession
    last_access: float


class SessionRegistry:
    """Thread-safe owner of all in-memory experiment sessions for one process."""

    def __init__(
        self,
        ttl_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a registry with a positive inactivity TTL and monotonic clock."""

        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise ValueError("ttl_seconds must be a finite positive number")
        self._ttl_seconds = float(ttl_seconds)
        if not math.isfinite(self._ttl_seconds) or self._ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a finite positive number")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, _RegistryEntry] = {}

    @property
    def ttl_seconds(self) -> float:
        """Return the configured inactivity lifetime in seconds."""

        return self._ttl_seconds

    def _expired_ids_locked(self, now: float) -> tuple[str, ...]:
        """Remove and cancel expired entries while the registry lock is held."""

        expired = tuple(
            session_id
            for session_id, entry in self._entries.items()
            if now - entry.last_access >= self._ttl_seconds
        )
        for session_id in expired:
            entry = self._entries.pop(session_id)
            entry.session.request_cancel()
        return expired

    def create(self, session_id: str | None = None) -> ExperimentSession:
        """Create one new session, rejecting a live duplicate ID."""

        identifier = _validated_session_id(session_id)
        now = self._clock()
        with self._lock:
            self._expired_ids_locked(now)
            if identifier in self._entries:
                raise ValueError(f"Session already exists: {identifier}")
            session = ExperimentSession(ExperimentState(session_id=identifier))
            self._entries[identifier] = _RegistryEntry(session, now)
            return session

    def get(self, session_id: str, *, touch: bool = True) -> ExperimentSession:
        """Return a live session and optionally extend its inactivity lifetime."""

        identifier = _validated_session_id(session_id)
        now = self._clock()
        with self._lock:
            self._expired_ids_locked(now)
            try:
                entry = self._entries[identifier]
            except KeyError as error:
                raise KeyError(f"Unknown or expired session: {identifier}") from error
            if touch:
                entry.last_access = now
            return entry.session

    def get_or_create(self, session_id: str | None = None) -> ExperimentSession:
        """Return an existing ID or atomically create it for a new browser tab."""

        identifier = _validated_session_id(session_id)
        now = self._clock()
        with self._lock:
            self._expired_ids_locked(now)
            entry = self._entries.get(identifier)
            if entry is None:
                session = ExperimentSession(ExperimentState(session_id=identifier))
                self._entries[identifier] = _RegistryEntry(session, now)
                return session
            entry.last_access = now
            return entry.session

    def touch(self, session_id: str) -> None:
        """Extend one session's TTL after user or worker activity."""

        self.get(session_id, touch=True)

    def cancel(self, session_id: str) -> bool:
        """Signal a live session's current cancellation token."""

        identifier = _validated_session_id(session_id)
        now = self._clock()
        with self._lock:
            self._expired_ids_locked(now)
            entry = self._entries.get(identifier)
            if entry is None:
                return False
            # Do not take the per-session state lock here: a long-running worker
            # may own it, and cooperative cancellation must remain immediate.
            entry.last_access = now
            entry.session.request_cancel()
            return True

    def reset(self, session_id: str) -> ExperimentSession:
        """Cancel and clear one session while preserving its stable browser ID."""

        identifier = _validated_session_id(session_id)
        now = self._clock()
        with self._lock:
            self._expired_ids_locked(now)
            try:
                entry = self._entries[identifier]
            except KeyError as error:
                raise KeyError(f"Unknown or expired session: {identifier}") from error
            entry.last_access = now
            session = entry.session

        # Never wait for a state lock while holding the registry lock. Workers
        # can safely report TTL activity while another caller resets the state.
        session.reset()
        with self._lock:
            if self._entries.get(identifier) is not entry:
                # Removal or expiry won the race; keep the detached object
                # cancelled rather than leaving its newly installed token live.
                session.request_cancel()
                raise KeyError(f"Session was removed while resetting: {identifier}")
            entry.last_access = self._clock()
        return session

    def remove(self, session_id: str) -> bool:
        """Remove one live session and signal any retained worker token."""

        identifier = _validated_session_id(session_id)
        with self._lock:
            entry = self._entries.pop(identifier, None)
            if entry is None:
                return False
            entry.session.request_cancel()
            return True

    def cleanup_expired(self) -> tuple[str, ...]:
        """Cancel and remove every session whose inactivity TTL elapsed."""

        with self._lock:
            return self._expired_ids_locked(self._clock())

    def close(self) -> None:
        """Cancel all sessions and release registry-owned state references."""

        with self._lock:
            entries = tuple(self._entries.values())
            self._entries.clear()
        for entry in entries:
            entry.session.request_cancel()

    def ids(self) -> tuple[str, ...]:
        """Return a stable snapshot of live IDs without extending their TTLs."""

        with self._lock:
            self._expired_ids_locked(self._clock())
            return tuple(sorted(self._entries))

    def __len__(self) -> int:
        """Return the number of currently live sessions."""

        return len(self.ids())


__all__ = ["ExperimentSession", "SessionRegistry"]
