"""SQLite storage + CSV exports + minute-level rollups.

Schema
------
frames(ts REAL, frame_idx INTEGER, n_vehicles INTEGER,
       mean_speed_px REAL, density REAL, congestion REAL)

events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT,
       class_name TEXT, direction TEXT, track_id INTEGER,
       payload_json TEXT)

minute_counts(minute_ts INTEGER, class_name TEXT, direction TEXT,
              count INTEGER, avg_congestion REAL,
              PRIMARY KEY(minute_ts, class_name, direction))
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from .events import Event
from .features import FrameFeatures

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    ts             REAL    NOT NULL,
    frame_idx      INTEGER NOT NULL,
    n_vehicles     INTEGER NOT NULL,
    mean_speed_px  REAL    NOT NULL,
    density        REAL    NOT NULL,
    congestion     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    type          TEXT    NOT NULL,
    class_name    TEXT,
    direction     TEXT,
    track_id      INTEGER,
    payload_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);

CREATE TABLE IF NOT EXISTS minute_counts (
    minute_ts       INTEGER NOT NULL,
    class_name      TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    count           INTEGER NOT NULL,
    avg_congestion  REAL    NOT NULL,
    PRIMARY KEY (minute_ts, class_name, direction)
);
CREATE INDEX IF NOT EXISTS idx_minute_counts_ts ON minute_counts(minute_ts);
"""


class Storage:
    def __init__(
        self,
        sqlite_path: str = "data/traffic.db",
        csv_dir: str = "data",
        rollup_seconds: int = 60,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.csv_dir = csv_dir
        self.rollup_seconds = int(rollup_seconds)
        Path(self.csv_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.sqlite_path) or ".").mkdir(
            parents=True, exist_ok=True,
        )

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.sqlite_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)

        self._events_csv = os.path.join(self.csv_dir, "events.csv")
        self._minute_csv = os.path.join(self.csv_dir, "minute_counts.csv")
        self._ensure_csv(
            self._events_csv,
            ["ts", "iso_ts", "type", "class_name",
             "direction", "track_id", "payload_json"],
        )
        self._ensure_csv(
            self._minute_csv,
            ["minute_ts", "iso_minute", "class_name",
             "direction", "count", "avg_congestion"],
        )

        self._scheduler = BackgroundScheduler(daemon=True)
        self._scheduler.add_job(
            self._rollup_minute,
            "interval",
            seconds=self.rollup_seconds,
            id="minute_rollup",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        log.info(
            "Storage ready: db=%s csv_dir=%s rollup=%ds",
            self.sqlite_path, self.csv_dir, self.rollup_seconds,
        )

    # ------------------------------------------------------------------ #
    # csv plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_csv(path: str, header: List[str]) -> None:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    @contextmanager
    def _csv_append(self, path: str):
        with open(path, "a", newline="", encoding="utf-8") as f:
            yield csv.writer(f)

    # ------------------------------------------------------------------ #
    # writers
    # ------------------------------------------------------------------ #
    def write_frame(self, frame_idx: int, ts: float, f: FrameFeatures) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO frames(ts, frame_idx, n_vehicles, "
                "mean_speed_px, density, congestion) VALUES (?,?,?,?,?,?)",
                (ts, int(frame_idx), int(f.n_vehicles),
                 float(f.mean_speed_px), float(f.density),
                 float(f.congestion)),
            )

    def write_events(self, events: Iterable[Event]) -> None:
        events = list(events)
        if not events:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO events(ts, type, class_name, direction, "
                "track_id, payload_json) VALUES (?,?,?,?,?,?)",
                [
                    (e.ts, e.type, e.class_name, e.direction,
                     e.track_id,
                     json.dumps(e.payload) if e.payload else None)
                    for e in events
                ],
            )
            with self._csv_append(self._events_csv) as w:
                for e in events:
                    w.writerow([
                        f"{e.ts:.3f}",
                        time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.ts),
                        ),
                        e.type,
                        e.class_name or "",
                        e.direction or "",
                        e.track_id if e.track_id is not None else "",
                        json.dumps(e.payload) if e.payload else "",
                    ])

    # ------------------------------------------------------------------ #
    # minute rollups
    # ------------------------------------------------------------------ #
    def _rollup_minute(self) -> None:
        """Aggregate the just-finished minute and append to CSV + table."""
        try:
            now = int(time.time())
            minute_end = (now // 60) * 60
            minute_start = minute_end - 60
            iso_minute = time.strftime(
                "%Y-%m-%dT%H:%M:00Z", time.gmtime(minute_start),
            )

            with self._lock:
                # Average congestion in the window.
                cur = self._conn.execute(
                    "SELECT AVG(congestion) FROM frames "
                    "WHERE ts >= ? AND ts < ?",
                    (minute_start, minute_end),
                )
                avg_cong = cur.fetchone()[0] or 0.0

                # LINE_CROSS counts grouped by class_name+direction.
                cur = self._conn.execute(
                    "SELECT COALESCE(class_name,''), COALESCE(direction,''), "
                    "       COUNT(*) "
                    "FROM events "
                    "WHERE type='LINE_CROSS' AND ts >= ? AND ts < ? "
                    "GROUP BY class_name, direction",
                    (minute_start, minute_end),
                )
                rows = cur.fetchall()

                if not rows:
                    # Still write an "all/all/0" row so the dashboard sees
                    # uptime.
                    rows = [("", "", 0)]

                params = [
                    (minute_start, cn, dr, int(cnt), float(avg_cong))
                    for (cn, dr, cnt) in rows
                ]
                self._conn.executemany(
                    "INSERT OR REPLACE INTO minute_counts "
                    "(minute_ts, class_name, direction, count, avg_congestion) "
                    "VALUES (?,?,?,?,?)",
                    params,
                )

                with self._csv_append(self._minute_csv) as w:
                    for (_ts, cn, dr, cnt, ac) in params:
                        w.writerow([
                            minute_start, iso_minute, cn, dr, cnt,
                            f"{ac:.4f}",
                        ])

            log.debug(
                "Minute rollup %s: %d rows, avg_cong=%.3f",
                iso_minute, len(params), avg_cong,
            )
        except Exception:  # noqa: BLE001
            log.exception("Minute rollup failed.")

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self._rollup_minute()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self._conn.close()
        log.info("Storage closed.")
