"""SQLite-backed generation history: one row per generated image, with the full
parameter set stored as JSON. Initialization is lazy so constructing a History at
import time (in the default app) touches no filesystem — the DB is created on first use."""
import json
import sqlite3
import time
from pathlib import Path
from typing import List


class History:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ready = False

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    seed INTEGER,
                    created_at TEXT NOT NULL,
                    params TEXT NOT NULL
                )"""
            )
        self._ready = True

    def add(self, job_id: str, url: str, mode: str, seed: int, params: dict) -> None:
        self._ensure()
        with self._conn() as c:
            c.execute(
                "INSERT INTO generations (job_id, url, mode, seed, created_at, params) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, url, mode, seed,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), json.dumps(params)),
            )

    def list(self, limit: int = 100) -> List[dict]:
        self._ensure()
        with self._conn() as c:
            rows = c.execute(
                "SELECT job_id, url, mode, seed, created_at, params "
                "FROM generations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"job_id": r["job_id"], "url": r["url"], "mode": r["mode"], "seed": r["seed"],
             "created_at": r["created_at"], "params": json.loads(r["params"])}
            for r in rows
        ]

    def delete(self, job_id: str) -> bool:
        self._ensure()
        with self._conn() as c:
            cur = c.execute("DELETE FROM generations WHERE job_id = ?", (job_id,))
            return cur.rowcount > 0
