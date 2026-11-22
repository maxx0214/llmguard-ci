"""SQLite cache and single-owner run checkpoints for normalized case observations."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path


def fingerprint(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _database(path: Path, stack: ExitStack, *, timeout: float = 5) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=timeout)
    stack.callback(connection.close)
    return connection


class CaseResults:
    def __init__(self, cache: Path | None, checkpoint: Path | None, run_fingerprint: str,
                 state_scope: Callable[[bytes], str] | None = None):
        self.cache_path = cache
        self.checkpoint_path = checkpoint
        self.run_fingerprint = run_fingerprint
        self.state_scope = state_scope
        self.cache_scope = None
        self.cache = None
        self.checkpoint = None
        self._stack = ExitStack()

    def __enter__(self) -> CaseResults:
        with ExitStack() as stack:
            if self.checkpoint_path is not None:
                lease = _database(self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".lease"), stack, timeout=0)
                try:
                    lease.execute("BEGIN EXCLUSIVE")
                except sqlite3.OperationalError as exc:
                    if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                        raise ValueError("Checkpoint is already open in another evaluation") from None
                    raise
                self.checkpoint = _database(self.checkpoint_path, stack)
                with self.checkpoint:
                    scope = self._scope(self.checkpoint)
                    run_fingerprint = fingerprint([self.run_fingerprint, scope]) if scope is not None else self.run_fingerprint
                    self.checkpoint.execute("CREATE TABLE IF NOT EXISTS run_config (id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL)")
                    self.checkpoint.execute("INSERT OR IGNORE INTO run_config VALUES (1, ?)", (run_fingerprint,))
                    actual = self.checkpoint.execute("SELECT fingerprint FROM run_config WHERE id = 1").fetchone()[0]
                    if actual != run_fingerprint:
                        raise ValueError("Checkpoint fingerprint does not match dataset, system or evaluation settings")
                    self.checkpoint.execute("CREATE TABLE IF NOT EXISTS completed_cases (key TEXT PRIMARY KEY, observation TEXT NOT NULL)")
            if self.cache_path is not None:
                self.cache = _database(self.cache_path, stack)
                with self.cache:
                    self.cache_scope = self._scope(self.cache)
                    self.cache.execute("CREATE TABLE IF NOT EXISTS cached_cases (key TEXT PRIMARY KEY, observation TEXT NOT NULL)")
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *_):
        self._stack.close()

    def _scope(self, connection):
        if self.state_scope is None:
            return None
        connection.execute("CREATE TABLE IF NOT EXISTS state_scope (id INTEGER PRIMARY KEY, secret BLOB NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO state_scope VALUES (1, ?)", (secrets.token_bytes(32),))
        secret = connection.execute("SELECT secret FROM state_scope WHERE id = 1").fetchone()[0]
        return self.state_scope(secret)

    def completed(self) -> dict:
        if self.checkpoint is None:
            return {}
        return {key: json.loads(data) for key, data in self.checkpoint.execute("SELECT key, observation FROM completed_cases")}

    def lookup(self, key: str):
        row = self.cache.execute("SELECT observation FROM cached_cases WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def remember(self, key: str, observation: dict):
        with self.cache:
            self.cache.execute("INSERT OR REPLACE INTO cached_cases VALUES (?, ?)", (key, json.dumps(observation, allow_nan=False)))

    def save(self, case_id: str, observation: dict):
        if self.checkpoint is not None:
            with self.checkpoint:
                self.checkpoint.execute("INSERT OR REPLACE INTO completed_cases VALUES (?, ?)",
                                        (case_id, json.dumps(observation, allow_nan=False)))
